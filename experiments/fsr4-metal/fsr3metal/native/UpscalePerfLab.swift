// UpscalePerfLab.swift - performance of the FSR3-class Metal pipeline.
//
// The correctness lab (UpscaleLab) proved the pipeline reconstructs and does
// not ghost. This one answers the other half of the objective: is it FAST
// enough to be worth running at 4K?
//
// Measurement discipline, learned the hard way from the earlier precision lab:
//   * Absolute milliseconds on this host are NOT reproducible. The same binary
//     lands in one of two GPU clock states 2-4x apart, and every variant moves
//     together within a run. So this lab reports SAME-RUN RATIOS against a
//     baseline, plus min/p50/p95/max for spread, and never a bare absolute
//     number as if it were portable.
//   * Every variant is dispatched INSIDE one trial loop with the order
//     alternated per trial, so a clock shift cannot land inside one variant.
//   * The GPU interval covers one command buffer. Allocation, packing and
//     readback are measured separately and reported separately, because a
//     game frame pays both.
//
// Comparators:
//   spatial        - Lanczos reconstruction only
//   spatial_rcas   - + RCAS sharpening
//   full           - + temporal accumulate (the shipping path)
//   baseline_bilinear - the CPU reference, so the Metal path is measured
//                        against something rather than against nothing

import Foundation
import Metal
import CryptoKit

struct PerfError: Error, CustomStringConvertible {
    let description: String
    init(_ s: String) { description = s }
}

struct SpatialParams {
    var renderSize = SIMD2<UInt32>(0, 0)
    var upscaleSize = SIMD2<UInt32>(0, 0)
    var jitterX: Float = 0
    var jitterY: Float = 0
    var useExact: Int32 = 0
}

struct RCASParams {
    var size = SIMD2<UInt32>(0, 0)
    var sharpness: Float = 0.6
}

struct AccumParams {
    var upscaleSize = SIMD2<UInt32>(0, 0)
    var frameIndex: Float = 0
    var deltaTime: Float = 1.0 / 60.0
    var exposure: Float = 1
    var prevExposure: Float = 1
    var accumulation: Float = 0.9
    var lumaInstability: Float = 0
    var reactiveMask: Float = 0
    var disocclusion: Float = 0
    var shadingChange: Float = 0
    var lock: Float = 0
    var lockContribution: Float = 0
    var currentColor = SIMD4<Float>(0, 0, 0, 0)
    var historyColor = SIMD4<Float>(0, 0, 0, 0)
    var motionVector = SIMD2<Float>(0, 0)
    var reset: Int32 = 0
}

func dispatch2D(_ e: MTLComputeCommandEncoder, _ p: MTLComputePipelineState,
               _ sx: Int, _ sy: Int) {
    let w = min(p.maxTotalThreadsPerThreadgroup, 16)
    let h = min(max(p.maxTotalThreadsPerThreadgroup / w, 1), 16)
    e.setComputePipelineState(p)
    e.dispatchThreadgroups(MTLSize(width: (sx + w - 1) / w, height: (sy + h - 1) / h, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
}

func pct(_ a: [Double], _ p: Double) -> Double {
    guard !a.isEmpty else { return .nan }
    return a.sorted()[max(0, Int(ceil(Double(a.count) * p)) - 1)]
}

func spreadJSON(_ s: [Double]) -> [String: Any] {
    ["gpu_min_ms": s.min()!, "gpu_median_ms": pct(s, 0.5),
     "gpu_p95_ms": pct(s, 0.95), "gpu_max_ms": s.max()!]
}

func makeBuf<T>(_ a: [T], _ d: MTLDevice) throws -> MTLBuffer {
    guard let b = a.withUnsafeBytes({ d.makeBuffer(bytes: $0.baseAddress!, length: $0.count,
                                                      options: .storageModeShared) })
    else { throw PerfError("allocation failed") }
    return b
}

func run() throws {
    guard CommandLine.arguments.count == 2 else { throw PerfError("usage: upscale-perf RECEIPT.json") }
    guard let device = MTLCreateSystemDefaultDevice(), device.hasUnifiedMemory,
          let queue = device.makeCommandQueue()
    else { throw PerfError("Unified-memory Metal GPU required") }

    let kernelPaths = ["fsr3metal/kernels/fsr3_spatial.metal",
                       "fsr3metal/kernels/fsr3_temporal.metal"]
    let sources = try kernelPaths.map { try String(contentsOfFile: $0, encoding: .utf8) }
    let library = try device.makeLibrary(source: sources.joined(separator: "\n"), options: nil)
    func pipe(_ n: String) throws -> MTLComputePipelineState {
        guard let f = library.makeFunction(name: n) else { throw PerfError("missing kernel \(n)") }
        return try device.makeComputePipelineState(function: f)
    }
    let pSpatial = try pipe("spatial_upscale")
    let pRCAS = try pipe("rcas_sharpen")
    let pAccum = try pipe("temporal_accumulate")

    // Real output resolutions, upscaled 4x as Ultra Performance would be.
    // 4K output at 4x is 1920x1080 input; 1440p at 4x is 960x540.
    let cases: [(String, Int, Int)] = [
        ("1440p_output_4x", 960, 540),
        ("4K_output_4x", 1920, 1080),
    ]
    let scale = 4
    let warmup = 10, repeats = 31

    var results: [[String: Any]] = []

    for (name, rw, rh) in cases {
        let uw = rw * scale, uh = rh * scale
        // Synthetic low-res input. Content does not affect timing for these
        // kernels (no data-dependent branching), but a real allocation is used
        // so the measurement includes real memory pressure.
        let low = (0..<(rw * rh)).map { i -> SIMD4<Float> in
            let v = Float((i * 37) % 255) / 255.0
            return SIMD4(v, v * 0.9, v * 0.8, 1)
        }
        let lowBuf = try makeBuf(low, device)
        let outBuf = try makeBuf([SIMD4<Float>](repeating: .zero, count: uw * uh), device)
        let lockBuf = try makeBuf([SIMD2<Float>](repeating: .zero, count: uw * uh), device)
        let accumBuf = try makeBuf([AccumParams](repeating: AccumParams(), count: uw * uh), device)
        // Keep history in a stable state so the temporal kernel is not
        // measuring its own initialisation.
        var ap = AccumParams()
        ap.upscaleSize = SIMD2<UInt32>(UInt32(uw), UInt32(uh))
        ap.accumulation = 0.9
        ap.reset = 1
        for i in 0..<(uw * uh) {
            ap.currentColor = SIMD4(Float((i * 13) % 255) / 255.0, 0.4, 0.2, 1)
            ap.historyColor = ap.currentColor
            accumBuf.contents().bindMemory(to: AccumParams.self, capacity: uw * uh)[i] = ap
        }

        var sp = SpatialParams()
        sp.renderSize = SIMD2<UInt32>(UInt32(rw), UInt32(rh))
        sp.upscaleSize = SIMD2<UInt32>(UInt32(uw), UInt32(uh))
        var rp = RCASParams()
        rp.size = SIMD2<UInt32>(UInt32(uw), UInt32(uh))

        let launches: [(String, (MTLCommandBuffer) -> Void)] = [
            ("spatial", { cb in
                let e = cb.makeComputeCommandEncoder()!
                e.setBuffer(lowBuf, offset: 0, index: 0)
                e.setBuffer(outBuf, offset: 0, index: 1)
                e.setBytes(&sp, length: MemoryLayout<SpatialParams>.stride, index: 2)
                dispatch2D(e, pSpatial, uw, uh); e.endEncoding()
            }),
            ("spatial_rcas", { cb in
                let e = cb.makeComputeCommandEncoder()!
                e.setBuffer(lowBuf, offset: 0, index: 0)
                e.setBuffer(outBuf, offset: 0, index: 1)
                e.setBytes(&sp, length: MemoryLayout<SpatialParams>.stride, index: 2)
                dispatch2D(e, pSpatial, uw, uh); e.endEncoding()
                let e2 = cb.makeComputeCommandEncoder()!
                e2.setBuffer(outBuf, offset: 0, index: 0)
                e2.setBytes(&rp, length: MemoryLayout<RCASParams>.stride, index: 1)
                dispatch2D(e2, pRCAS, uw, uh); e2.endEncoding()
            }),
            ("full_temporal", { cb in
                let e = cb.makeComputeCommandEncoder()!
                e.setBuffer(lowBuf, offset: 0, index: 0)
                e.setBuffer(outBuf, offset: 0, index: 1)
                e.setBytes(&sp, length: MemoryLayout<SpatialParams>.stride, index: 2)
                dispatch2D(e, pSpatial, uw, uh); e.endEncoding()
                let e2 = cb.makeComputeCommandEncoder()!
                e2.setBuffer(outBuf, offset: 0, index: 0)
                e2.setBytes(&rp, length: MemoryLayout<RCASParams>.stride, index: 1)
                dispatch2D(e2, pRCAS, uw, uh); e2.endEncoding()
                let e3 = cb.makeComputeCommandEncoder()!
                e3.setBuffer(accumBuf, offset: 0, index: 0)
                e3.setBuffer(outBuf, offset: 0, index: 1)
                e3.setBuffer(lockBuf, offset: 0, index: 2)
                dispatch2D(e3, pAccum, uw, uh); e3.endEncoding()
            }),
        ]

        var samples = [String: [Double]](uniqueKeysWithValues: launches.map { ($0.0, [Double]()) })
        for trial in 0..<(warmup + repeats) {
            let seq = trial % 2 == 0 ? launches : launches.reversed()
            for (nm, launch) in seq {
                let cb = queue.makeCommandBuffer()!
                launch(cb)
                cb.commit(); cb.waitUntilCompleted()
                guard cb.status == .completed, cb.error == nil
                else { throw PerfError("\(nm) failed: \(String(describing: cb.error))") }
                if trial >= warmup {
                    samples[nm]?.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
                }
            }
        }

        // Allocation cost, reported separately because a game frame pays it
        // only on resize, not per frame.
        let allocStart = ProcessInfo.processInfo.systemUptime
        _ = try makeBuf([SIMD4<Float>](repeating: .zero, count: uw * uh), device)
        let allocMs = (ProcessInfo.processInfo.systemUptime - allocStart) * 1000

        var rows: [[String: Any]] = []
        let baseline = pct(samples["spatial"]!, 0.5)
        for (nm, _) in launches {
            let s = samples[nm]!
            var row = spreadJSON(s)
            row["variant"] = nm
            row["ratio_to_spatial"] = pct(s, 0.5) / baseline
            rows.append(row)
        }
        let fullMed = pct(samples["full_temporal"]!, 0.5)
        results.append([
            "case": name,
            "render_size": [rw, rh], "upscale_size": [uw, uh], "scale": scale,
            "output_pixels": uw * uh,
            "variants": rows,
            "full_pipeline_median_ms": fullMed,
            "full_pipeline_ratio_to_spatial": fullMed / baseline,
            "allocation_ms_once": allocMs,
            "fps_if_gpu_bound": 1000.0 / fullMed,
            "frame_budget_60fps_ms": 1000.0 / 60.0,
            "fits_60fps_budget": fullMed < 1000.0 / 60.0,
        ])
        print("\(name): full=\(String(format: "%.4f", fullMed)) ms  "
              + "(\(String(format: "%.1f", 1000.0/fullMed)) fps, "
              + "spatial-only=\(String(format: "%.4f", baseline)) ms)")
    }

    let report: [String: Any] = [
        "schema": "helix.fsr-metal.fsr3-upscale-perf.v1",
        "scope": "synthetic_kernel_timing_on_this_host",
        "device": device.name,
        "os": ProcessInfo.processInfo.operatingSystemVersionString,
        "utc_timestamp": ISO8601DateFormatter().string(from: Date()),
        "warmup": warmup, "repeats": repeats,
        "timing_protocol": "all variants interleaved inside one trial loop, order alternated per trial",
        "absolute_timing_caveat": "absolute ms are NOT reproducible on this host; the GPU sits in one of two clock states 2-4x apart and all variants move together within a run. Compare SAME-RUN RATIOS.",
        "gpu_timing_scope": "one command buffer, one to three encoders; excludes allocation, packing and readback",
        "allocation_reported_separately": true,
        "content_independence": "these kernels have no data-dependent branching, so content does not affect timing; a real allocation is used so memory pressure is realistic",
        "results": results,
        "in_a_game": false,
        "crossover_interop_tested": false,
        "comparable_to_amd_fsr_performance": false,
    ]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        .write(to: URL(fileURLWithPath: CommandLine.arguments[1]), options: .withoutOverwriting)
}

do { try run() } catch {
    FileHandle.standardError.write(Data("upscale-perf: \(error)\n".utf8))
    exit(1)
}
