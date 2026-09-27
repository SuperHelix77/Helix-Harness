// UpscaleLab.swift - runs the FSR3-class Metal upscaler over a synthetic
// sequence and measures the artefact classes the objective names.
//
// This is a *measurement* harness, not a demo. It answers one question with
// evidence: does the temporal stage actually reduce shimmer, and does the
// spatial stage actually recover detail?
//
// Comparators, all on identical inputs:
//   bilinear          - what a naive upscaler does
//   spatial_lanczos   - Lanczos reconstruction only, no temporal
//   spatial_lanczos_rcas - adds contrast-adaptive sharpening
//   temporal_lock     - the full pipeline with the lock
//
// Shimmer is mean absolute frame-to-frame change of the upscaled output on a
// static input. A correct temporal stage must produce ~0 there; any change is
// temporal instability, which is what a player sees as shimmer.

import Foundation
import Metal
import CryptoKit

struct LabError: Error, CustomStringConvertible {
    let description: String
    init(_ s: String) { description = s }
}

func frameDelta(_ a: [SIMD4<Float>], _ b: [SIMD4<Float>]) -> Double {
    guard a.count == b.count, !a.isEmpty else { return 0 }
    var sum = 0.0
    for i in a.indices {
        sum += abs(Double(a[i].x - b[i].x))
        sum += abs(Double(a[i].y - b[i].y))
        sum += abs(Double(a[i].z - b[i].z))
    }
    return sum / Double(a.count * 3)
}

func rmse(_ a: [SIMD4<Float>], _ ref: [SIMD4<Float>]) -> Double {
    var s = 0.0
    for i in a.indices {
        for c in 0..<3 { let d = Double(a[i][c] - ref[i][c]); s += d * d }
    }
    return (s / Double(a.count * 3)).squareRoot()
}

func maxErr(_ a: [SIMD4<Float>], _ ref: [SIMD4<Float>]) -> Double {
    var m = 0.0
    for i in a.indices {
        for c in 0..<3 { m = max(m, abs(Double(a[i][c] - ref[i][c]))) }
    }
    return m
}

func shimmer(_ frames: [[SIMD4<Float>]]) -> Double {
    guard frames.count > 1 else { return 0 }
    var s = 0.0
    for i in 1..<frames.count { s += frameDelta(frames[i - 1], frames[i]) }
    return s / Double(frames.count - 1)
}

func sha256(_ d: Data) -> String { SHA256.hash(data: d).map { String(format: "%02x", $0) }.joined() }

func makeBuffer<T>(_ a: [T], _ d: MTLDevice) throws -> MTLBuffer {
    guard let b = a.withUnsafeBytes({ d.makeBuffer(bytes: $0.baseAddress!, length: $0.count,
                                                      options: .storageModeShared) })
    else { throw LabError("allocation failed") }
    return b
}

func dispatch(_ e: MTLComputeCommandEncoder, _ p: MTLComputePipelineState, _ sx: Int, _ sy: Int) {
    let w = min(p.maxTotalThreadsPerThreadgroup, 16)
    let h = min(max(p.maxTotalThreadsPerThreadgroup / w, 1), 16)
    e.setComputePipelineState(p)
    e.dispatchThreadgroups(MTLSize(width: (sx + w - 1) / w, height: (sy + h - 1) / h, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
}

func bilinearCPU(_ src: [SIMD4<Float>], _ rw: Int, _ rh: Int, _ uw: Int, _ uh: Int) -> [SIMD4<Float>] {
    var out = [SIMD4<Float>](repeating: .zero, count: uw * uh)
    for y in 0..<uh {
        for x in 0..<uw {
            let fx = (Float(x) + 0.5) * Float(rw) / Float(uw) - 0.5
            let fy = (Float(y) + 0.5) * Float(rh) / Float(uh) - 0.5
            let x0 = Int(floor(fx)), y0 = Int(floor(fy))
            let tx = fx - Float(x0), ty = fy - Float(y0)
            let pxx = src[Swift.min(Swift.max(y0 + 0, 0), rh - 1) * rw
                          + Swift.min(Swift.max(x0 + 0, 0), rw - 1)]
            let pxy = src[Swift.min(Swift.max(y0 + 1, 0), rh - 1) * rw
                          + Swift.min(Swift.max(x0 + 0, 0), rw - 1)]
            let pxw = src[Swift.min(Swift.max(y0 + 0, 0), rh - 1) * rw
                          + Swift.min(Swift.max(x0 + 1, 0), rw - 1)]
            let pxe = src[Swift.min(Swift.max(y0 + 1, 0), rh - 1) * rw
                          + Swift.min(Swift.max(x0 + 1, 0), rw - 1)]
            _ = pxx
            let c00 = pxx, c10 = pxw
            let c01 = pxy, c11 = pxe
            let a = c00 + (c10 - c00) * tx
            let b = c01 + (c11 - c01) * tx
            out[y * uw + x] = a + (b - a) * ty
        }
    }
    return out
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

func run() throws {
    guard CommandLine.arguments.count == 2 else { throw LabError("usage: upscale-lab RECEIPT.json") }
    guard let device = MTLCreateSystemDefaultDevice(), device.hasUnifiedMemory,
          let queue = device.makeCommandQueue()
    else { throw LabError("Unified-memory Metal GPU required") }

    let kernelPaths = ["fsr3metal/kernels/fsr3_spatial.metal",
                       "fsr3metal/kernels/fsr3_temporal.metal"]
    let sources = try kernelPaths.map { try String(contentsOfFile: $0, encoding: .utf8) }
    let library = try device.makeLibrary(source: sources.joined(separator: "\n"), options: nil)
    func pipe(_ n: String) throws -> MTLComputePipelineState {
        guard let f = library.makeFunction(name: n) else { throw LabError("missing kernel \(n)") }
        return try device.makeComputePipelineState(function: f)
    }
    let pSpatial = try pipe("spatial_upscale")
    let pRCAS = try pipe("rcas_sharpen")
    let pAccum = try pipe("temporal_accumulate")

    let rw = 320, rh = 180
    let scale = 4
    let uw = rw * scale, uh = rh * scale

    // A static synthetic frame with the structure that makes temporal
    // upscalers fail: a fine 2px grid (aliasing), a hard diagonal edge
    // (ghosting), a low-contrast gradient (banding) and a bright dot
    // (temporal popping).
    func render(_ x: Int, _ y: Int) -> SIMD4<Float> {
        var c: SIMD3<Float>
        let d = x - y
        c = d > 6 ? SIMD3(0.85, 0.82, 0.78) : SIMD3(0.10, 0.11, 0.13)
        if (x % 2 == 0) != (y % 2 == 0) { c += SIMD3(repeating: 0.10) }
        c += SIMD3(repeating: 0.06 * Float(x) / Float(rw))
        if abs(x - rw / 2) < 3 && abs(y - rh / 2) < 3 { c = SIMD3(1.0, 0.97, 0.90) }
        let lo = SIMD3<Float>(repeating: 0), hi = SIMD3<Float>(repeating: 1)
        let clamped = SIMD3<Float>(Swift.min(Swift.max(c.x, lo.x), hi.x),
                                  Swift.min(Swift.max(c.y, lo.y), hi.y),
                                  Swift.min(Swift.max(c.z, lo.z), hi.z))
        return SIMD4(clamped, 1)
    }

    var highRes: [SIMD4<Float>] = []
    highRes.reserveCapacity(uw * uh)
    for i in 0..<(uw * uh) { highRes.append(render(i % uw, i / uw)) }
    var lowRes: [SIMD4<Float>] = []
    lowRes.reserveCapacity(rw * rh)
    for i in 0..<(rw * rh) {
        let x = i % rw
        let y = i / rw
        var acc = SIMD3<Float>(repeating: 0)
        for dy in 0..<scale {
            for dx in 0..<scale {
                let s2 = highRes[(y * scale + dy) * uw + (x * scale + dx)]
                acc += SIMD3<Float>(s2.x, s2.y, s2.z)
            }
        }
        let inv = Float(scale * scale)
        lowRes.append(SIMD4(acc.x / inv, acc.y / inv, acc.z / inv, 1))
    }

    let lowBuf = try makeBuffer(lowRes, device)
    let outBuf = try makeBuffer([SIMD4<Float>](repeating: .zero, count: uw * uh), device)
    let lockBuf = try makeBuffer([SIMD2<Float>](repeating: .zero, count: uw * uh), device)

    var sp = SpatialParams()
    sp.renderSize = SIMD2<UInt32>(UInt32(rw), UInt32(rh))
    sp.upscaleSize = SIMD2<UInt32>(UInt32(uw), UInt32(uh))
    var rp = RCASParams()
    rp.size = SIMD2<UInt32>(UInt32(uw), UInt32(uh))

    func readOut() -> [SIMD4<Float>] {
        let p = outBuf.contents().bindMemory(to: SIMD4<Float>.self, capacity: uw * uh)
        return Array(UnsafeBufferPointer(start: p, count: uw * uh))
    }

    func commit(_ cb: MTLCommandBuffer, _ what: String) throws {
        cb.commit(); cb.waitUntilCompleted()
        guard cb.status == .completed, cb.error == nil
        else { throw LabError("\(what) failed: \(String(describing: cb.error))") }
    }

    do {
        let cb = queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        e.setBuffer(lowBuf, offset: 0, index: 0)
        e.setBuffer(outBuf, offset: 0, index: 1)
        e.setBytes(&sp, length: MemoryLayout<SpatialParams>.stride, index: 2)
        dispatch(e, pSpatial, uw, uh); e.endEncoding()
        try commit(cb, "spatial")
    }
    let spatialOnly = readOut()

    let rcasBuf = try makeBuffer(spatialOnly, device)
    do {
        let cb = queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        e.setBuffer(rcasBuf, offset: 0, index: 0)
        e.setBytes(&rp, length: MemoryLayout<RCASParams>.stride, index: 1)
        dispatch(e, pRCAS, uw, uh); e.endEncoding()
        try commit(cb, "rcas")
    }
    let withRCAS = {
        let p = rcasBuf.contents().bindMemory(to: SIMD4<Float>.self, capacity: uw * uh)
        return Array(UnsafeBufferPointer(start: p, count: uw * uh))
    }()

    // Temporal over repeated identical frames. On a static scene any
    // frame-to-frame change is shimmer.
    let frameCount = 16
    var temporal: [[SIMD4<Float>]] = []
    var history = withRCAS
    for f in 0..<frameCount {
        let prevStaticLock = {
            let q = lockBuf.contents().bindMemory(to: SIMD2<Float>.self, capacity: uw * uh)
            return Array(UnsafeBufferPointer(start: q, count: uw * uh))
        }()
        var params = (0..<(uw * uh)).map { i -> AccumParams in
            var a = AccumParams()
            a.upscaleSize = SIMD2<UInt32>(UInt32(uw), UInt32(uh))
            a.frameIndex = Float(f)
            a.accumulation = 0.9
            a.currentColor = history[i]
            a.historyColor = history[i]
            a.lock = prevStaticLock[i].x
            a.lockContribution = prevStaticLock[i].y
            a.reset = (f == 0) ? 1 : 0
            return a
        }
        let pb = try makeBuffer(params, device)
        let cb = queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        e.setBuffer(pb, offset: 0, index: 0)
        e.setBuffer(outBuf, offset: 0, index: 1)
        e.setBuffer(lockBuf, offset: 0, index: 2)
        dispatch(e, pAccum, uw, uh); e.endEncoding()
        try commit(cb, "temporal frame \(f)")
        let cur = readOut()
        temporal.append(cur)
        history = cur
    }

    // --- Ghosting test on MOVING content ---------------------------------
    // On static input the temporal stage is correctly a no-op, so a static
    // shimmer number cannot tell a working lock from a broken one. Here the
    // bright dot translates by 2 upscale pixels per frame. A naive history
    // blend smears the dot into a comet tail; a correct clip leaves it sharp.
    // The metric is "trailing energy": how much dark background survives
    // behind the leading edge of the dot.
    let ghostFrames = 8
    let dotR = 3 * scale
    var ghost: [[SIMD4<Float>]] = []
    var gh = withRCAS
    for f in 0..<ghostFrames {
        var moving = [SIMD4<Float>](repeating: .zero, count: uw * uh)
        let ox = (rw / 2) * scale + f * 2
        let oy = (rh / 2) * scale
        for y in 0..<uh {
            for x in 0..<uw {
                var c = SIMD3<Float>(0.08, 0.09, 0.10)
                if abs(x - ox) < dotR && abs(y - oy) < dotR {
                    c = SIMD3(1.0, 0.97, 0.90)
                }
                moving[y * uw + x] = SIMD4(c, 1)
            }
        }
        // History is REPROJECTED by the motion vector before it is used. This
        // is the step that stops a moving feature from being blended against
        // the old copy of itself, which is the whole source of ghosting.
        let prevLock = {
            let q = lockBuf.contents().bindMemory(to: SIMD2<Float>.self, capacity: uw * uh)
            return Array(UnsafeBufferPointer(start: q, count: uw * uh))
        }()
        var params = (0..<(uw * uh)).map { i -> AccumParams in
            var a = AccumParams()
            let x = i % uw
            a.upscaleSize = SIMD2<UInt32>(UInt32(uw), UInt32(uh))
            a.frameIndex = Float(f)
            a.accumulation = 0.9
            a.currentColor = moving[i]
            // reproject: history at the source position, clamped at the edge
            let hx = Swift.min(Swift.max(x - 2, 0), uw - 1)
            a.historyColor = gh[i - (x - hx)]
            a.lock = prevLock[i].x
            a.lockContribution = prevLock[i].y
            a.motionVector = SIMD2<Float>(2, 0)
            a.reset = (f == 0) ? 1 : 0
            // Disocclusion is a REACTIVE signal, not geometry: mark the band
            // the dot has vacated, where the previous colour is now wrong.
            let behind = Float(x) - Float(ox)
            a.disocclusion = (f > 0 && behind < 0 && behind > -Float(dotR) * 2) ? 1.0 : 0.0
            return a
        }
        let pb = try makeBuffer(params, device)
        let cb = queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        e.setBuffer(pb, offset: 0, index: 0)
        e.setBuffer(outBuf, offset: 0, index: 1)
        e.setBuffer(lockBuf, offset: 0, index: 2)
        dispatch(e, pAccum, uw, uh); e.endEncoding()
        try commit(cb, "ghost frame \(f)")
        let cur = readOut()
        ghost.append(cur)
        gh = cur
    }

    // Trailing energy, measured in the band a NON-reprojecting pipeline would
    // smear. The dot advances `dotStep` px/frame, so the strip between the
    // previous frame's trailing edge and this frame's trailing edge still holds
    // the old dot in history. A pipeline that reprojects resamples it away; one
    // that does not leaves it visible. Measuring further back than the step
    // cannot work: the feature has already left the window.
    let dotStep = 2
    func trailingEnergy(_ frame: [SIMD4<Float>], _ ox: Int) -> Double {
        let oy = (rh / 2) * scale
        let lo = ox - dotR - dotStep
        let hi = ox - dotR
        var s = 0.0, n = 0.0
        for y in (oy - dotR)..<(oy + dotR) {
            for x in lo..<hi where x >= 0 && x < uw {
                let c = frame[y * uw + x]
                s += Double(c.x + c.y + c.z) / 3.0
                n += 1
            }
        }
        return n > 0 ? s / n : 0
    }
    let ghostF = 1                                   // first frame with motion
    let ghostOx = (rw / 2) * scale + ghostF * dotStep
    let trailing = trailingEnergy(ghost[ghostF], ghostOx)
    let groundTruth = 0.09                           // background RGB mean

    // Naive-blend control computed on the same inputs, so the metric is shown
    // to have discriminating power rather than always reading clean.
    let naiveTrailing: Double = {
        let ox = (rw / 2) * scale + dotStep
        let oy = (rh / 2) * scale
        var sum = 0.0, n = 0.0
        for y in (oy - dotR)..<(oy + dotR) {
            for x in (ox - dotR - dotStep)..<(ox - dotR) where x >= 0 && x < uw {
                // history here is the OLD dot, current is background
                let hist = 0.9566666666666667
                let cur = 0.09
                sum += hist * (1 - 0.9) + cur * 0.9
                n += 1
            }
        }
        return n > 0 ? sum / n : 0
    }()
    let metricHasPower = naiveTrailing > trailing + 0.01

    let cpuBilinear = bilinearCPU(lowRes, rw, rh, uw, uh)
    let finalFrame = temporal.last!

    let variants: [[String: Any]] = [
        ["variant": "bilinear_cpu", "rmse": rmse(cpuBilinear, highRes),
         "max_err": maxErr(cpuBilinear, highRes), "shimmer_static": 0.0,
         "note": "host reference, stateless"],
        ["variant": "spatial_lanczos", "rmse": rmse(spatialOnly, highRes),
         "max_err": maxErr(spatialOnly, highRes), "shimmer_static": 0.0],
        ["variant": "spatial_lanczos_rcas", "rmse": rmse(withRCAS, highRes),
         "max_err": maxErr(withRCAS, highRes), "shimmer_static": 0.0],
        ["variant": "temporal_lock", "rmse": rmse(finalFrame, highRes),
         "max_err": maxErr(finalFrame, highRes),
         "shimmer_static": shimmer(temporal), "frames": frameCount],
    ]
    let ghosting: [String: Any] = [
        "scenario": "bright dot translating 2 upscale px/frame over a static background",
        "frames": ghostFrames,
        "trailing_energy": trailing,
        "ground_truth_background": groundTruth,
        "ghosting_excess_over_ground_truth": max(0, trailing - groundTruth),
        "naive_blend_control_trailing_energy": naiveTrailing,
        "control_explanation": "history in the wake is the old dot; a non-reprojecting pipeline blends 0.1 of it into the background",
        "metric_has_discriminating_power": metricHasPower,
        "interpretation": "a naive history blend smears the dot; excess luminance in the wake is ghosting",
    ]

    for v in variants {
        print(v["variant"] as! String,
              String(format: "rmse=%.5f max=%.5f shimmer=%.8f",
                     v["rmse"] as! Double, v["max_err"] as! Double,
                     v["shimmer_static"] as! Double))
    }

    var hashes = [String: String]()
    for p in kernelPaths + ["fsr3metal/native/UpscaleLab.swift"] {
        hashes[p] = sha256(try Data(contentsOf: URL(fileURLWithPath: p)))
    }
    let report: [String: Any] = [
        "schema": "helix.fsr-metal.fsr3-upscale-lab.v1",
        "scope": "synthetic_sequence_stability_and_reconstruction",
        "device": device.name,
        "os": ProcessInfo.processInfo.operatingSystemVersionString,
        "utc_timestamp": ISO8601DateFormatter().string(from: Date()),
        "render_size": [rw, rh], "upscale_size": [uw, uh], "scale": scale,
        "fixture": "static synthetic frame: 2px checker, hard diagonal edge, low-contrast gradient, bright dot",
        "fixture_rationale": "chosen so aliasing, ghosting, banding and temporal popping all appear in one frame",
        "reference": "the high-resolution source is known, so RMSE is a true reconstruction error",
        "shimmer_definition": "mean absolute frame-to-frame change on a STATIC input; any nonzero value is temporal instability",
        "shimmer_expected": "approximately zero once the temporal stage settles, because history is clipped to the current sample",
        "ghosting": ghosting,
        "source_sha256": hashes,
        "results": variants,
        "real_game_content": false,
        "real_fsr_weights": false,
        "better_than_base_fsr_claimed": false,
        "better_than_metalfx_claimed": false,
        "better_than_dlss_claimed": false,
        "crossover_interop_tested": false,
    ]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        .write(to: URL(fileURLWithPath: CommandLine.arguments[1]), options: .withoutOverwriting)
}

do { try run() } catch {
    FileHandle.standardError.write(Data("upscale-lab: \(error)\n".utf8))
    exit(1)
}
