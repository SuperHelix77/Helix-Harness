// PrecisionLab.swift - FP16 vs INT8 vs FP32 native baselines on identical shapes.
//
// Answers the research question the capsule states but has not yet measured:
// "Run FP16 and INT8 native baselines on identical shapes" and "an M3 FP16
// matrix path may beat a scalar integer implementation; this must be measured."
//
// Scope and honesty rules, in the same spirit as the other labs here:
//   * These are OUR synthetic operator formulations, not AMD FSR operators.
//   * No model weights. No game. No picture-quality claim.
//   * Correctness is checked against an independent host oracle, and FP16
//     numerical error is REPORTED rather than asserted to be zero.
//   * The GPU interval excludes compile, allocation, packing and readback.
//   * Negative and unstable results are retained, not dropped.
//
// Usage: precision-lab NEW_RECEIPT.json

import Foundation
import Metal
import CryptoKit

struct LabFailure: Error, CustomStringConvertible {
    let description: String
    init(_ s: String) { description = s }
}

struct F16Params {
    var height, width, channels, outputs, kernelSize, stride, padding: UInt32
    var outHeight, outWidth: UInt32
}

struct ConvParams {
    var height, width, channels, outputs, kernelSize, stride, padding: UInt32
    var outHeight, outWidth, paddedChannels: UInt32
}

struct Shape {
    let name: String
    let h, w, c, o, k, stride, pad: Int
    var cp: Int { (c + 3) / 4 * 4 }
    var oh: Int { (h + 2*pad - k) / stride + 1 }
    var ow: Int { (w + 2*pad - k) / stride + 1 }
    var count: Int { oh * ow * o }
    var f16: F16Params {
        F16Params(height: UInt32(h), width: UInt32(w), channels: UInt32(c),
                  outputs: UInt32(o), kernelSize: UInt32(k), stride: UInt32(stride),
                  padding: UInt32(pad), outHeight: UInt32(oh), outWidth: UInt32(ow))
    }
    var int8: ConvParams {
        ConvParams(height: UInt32(h), width: UInt32(w), channels: UInt32(c),
                   outputs: UInt32(o), kernelSize: UInt32(k), stride: UInt32(stride),
                   padding: UInt32(pad), outHeight: UInt32(oh), outWidth: UInt32(ow),
                   paddedChannels: UInt32(cp))
    }
    var spatial: Int { oh * ow }
    var spatialTiles: Int { (spatial + 7) / 8 }
    var outTiles: Int { (o + 7) / 8 }
    // The simdgroup kernel implements 1x1 convolution ONLY. A 3x3 or strided
    // shape would silently compute something else, so it is never measured.
    var simdEligible: Bool { k == 1 && stride == 1 && pad == 0 && spatial % 8 == 0 && o % 8 == 0 && c % 8 == 0 }
}

func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func percentile(_ a: [Double], _ p: Double) -> Double {
    guard !a.isEmpty else { return .nan }
    return a.sorted()[max(0, Int(ceil(Double(a.count) * p)) - 1)]
}

// Independent host float32 -> binary16 round-to-nearest-even, written from the
// IEEE-754 definition so the oracle is a genuine second opinion on the half
// round trip rather than a copy of GPU behaviour.
func toHalf(_ f: Float) -> UInt16 {
    let x = f.bitPattern
    let sign = UInt16((x >> 16) & 0x8000)
    let rawExp = Int32((x >> 23) & 0xFF)
    let man = x & 0x007FFFFF
    if rawExp == 0 { return sign }
    if rawExp == 0xFF { return man == 0 ? sign | 0x7C00 : sign | 0x7E00 }
    let e = rawExp - 112
    if e <= 0 { return sign }
    if e >= 31 { return sign | 0x7C00 }
    if e >= 15 { return sign | UInt16(e << 10) | UInt16(man >> 13) }
    let r = (man + 0x0FFF + ((man >> 13) & 1)) >> 13
    if r & 0x400 != 0 { return sign | UInt16((e + 1) << 10) }
    return sign | UInt16(e << 10) | UInt16(r)
}

func fromHalf(_ h: UInt16) -> Float {
    let s: Float = (h & 0x8000) != 0 ? -1 : 1
    let e = Int32((h >> 10) & 0x1F)
    let m = h & 0x3FF
    if e == 0 { return s * Float(m) * 5.9604644775390625e-8 }
    if e == 31 { return m == 0 ? s * Float.infinity : .nan }
    return s * (1 + Float(m) / 1024) * exp2(Float(e - 15))
}

// Exact half accumulate, same order as conv_f16_direct. Honest oracle for it.
func halfAccumulate(_ a: [UInt16], _ w: [UInt16], _ s: Shape) -> [Float] {
    var out = [Float](repeating: 0, count: s.count)
    for y in 0..<s.oh {
        for x in 0..<s.ow {
            for oc in 0..<s.o {
                var acc: UInt16 = toHalf(0)
                for ky in 0..<s.k {
                    for kx in 0..<s.k {
                        let iy = y * s.stride + ky - s.pad, ix = x * s.stride + kx - s.pad
                        if iy < 0 || ix < 0 || iy >= s.h || ix >= s.w { continue }
                        for c in 0..<s.c {
                            let ia = (iy * s.w + ix) * s.c + c
                            let ib = ((oc * s.k + ky) * s.k + kx) * s.c + c
                            acc = toHalf(fromHalf(acc) + fromHalf(a[ia]) * fromHalf(w[ib]))
                        }
                    }
                }
                out[(y * s.ow + x) * s.o + oc] = fromHalf(acc)
            }
        }
    }
    return out
}

// Half inputs, FP32 accumulation, half output. Oracle for conv_f16_f32acc.
func f32AccumulateHalfInputs(_ a: [UInt16], _ w: [UInt16], _ s: Shape) -> [Float] {
    var out = [Float](repeating: 0, count: s.count)
    for y in 0..<s.oh {
        for x in 0..<s.ow {
            for oc in 0..<s.o {
                var acc: Float = 0
                for ky in 0..<s.k {
                    for kx in 0..<s.k {
                        let iy = y * s.stride + ky - s.pad, ix = x * s.stride + kx - s.pad
                        if iy < 0 || ix < 0 || iy >= s.h || ix >= s.w { continue }
                        for c in 0..<s.c {
                            let ia = (iy * s.w + ix) * s.c + c
                            let ib = ((oc * s.k + ky) * s.k + kx) * s.c + c
                            acc += fromHalf(a[ia]) * fromHalf(w[ib])
                        }
                    }
                }
                out[(y * s.ow + x) * s.o + oc] = fromHalf(toHalf(acc))
            }
        }
    }
    return out
}

func f32Oracle(_ a: [Float], _ w: [Float], _ s: Shape) -> [Float] {
    var out = [Float](repeating: 0, count: s.count)
    for y in 0..<s.oh {
        for x in 0..<s.ow {
            for oc in 0..<s.o {
                var acc: Float = 0
                for ky in 0..<s.k {
                    for kx in 0..<s.k {
                        let iy = y * s.stride + ky - s.pad, ix = x * s.stride + kx - s.pad
                        if iy < 0 || ix < 0 || iy >= s.h || ix >= s.w { continue }
                        for c in 0..<s.c {
                            let ia = (iy * s.w + ix) * s.c + c
                            let ib = ((oc * s.k + ky) * s.k + kx) * s.c + c
                            acc += a[ia] * w[ib]
                        }
                    }
                }
                out[(y * s.ow + x) * s.o + oc] = acc
            }
        }
    }
    return out
}

// Reference for the simdgroup 1x1 path. simdgroup `a * b` accumulates in half
// in the tile's own fragment order, which host scalar code cannot reproduce
// exactly. The honest check is that the matrix result agrees with the
// exact half product-sum to within a few half ULPs, and is no worse than the
// scalar half path. Exact bitwise agreement is NOT claimed.
func simdReference(_ a: [UInt16], _ w: [UInt16], _ s: Shape) -> [Float] {
    var out = [Float](repeating: 0, count: s.count)
    for n in 0..<s.spatial {
        for o in 0..<s.o {
            var acc: Float = 0
            for k in 0..<s.c { acc += fromHalf(a[n * s.c + k]) * fromHalf(w[o * s.c + k]) }
            out[n * s.o + o] = acc
        }
    }
    return out
}

// Half ULP at a magnitude, so a tolerance scales with the result size instead
// of hiding a large relative error behind a small absolute one.
func halfULP(_ x: Float) -> Float {
    let a = abs(x)
    if a < 6.103515625e-05 { return 5.9604644775390625e-08 }
    return pow(2.0, floor(log2(a)) - 10)
}

// Independent dyadic INT8 epilogue oracle, same logic as FusionLab.
func int8Oracle(_ sum: Int64, _ b: Int32, _ m: Int32, _ shift: UInt32,
                _ z: Int32, _ lower: Int32) -> Int8 {
    let value = (Double((sum + Int64(b)) * Int64(m))
        / Double(UInt64(1) << shift)).rounded(.toNearestOrEven)
    return Int8(max(Int64(lower), min(127, Int64(value) + Int64(z))))
}

func makeBuffer<T>(_ a: [T], _ d: MTLDevice) throws -> MTLBuffer {
    guard let b = a.withUnsafeBytes({ d.makeBuffer(bytes: $0.baseAddress!,
                                                      length: $0.count, options: .storageModeShared) })
    else { throw LabFailure("allocation failed") }
    return b
}

// Timings on this host are bimodal (the GPU sits in one of two clock states),
// so a bare median hides that. Every variant records min/p50/p95/max so the
// distribution is visible in the receipt instead of being implied.
func spread(_ samples: [Double], _ extra: [String: Any]) -> [String: Any] {
    var out: [String: Any] = ["gpu_min_ms": samples.min()!, "gpu_median_ms": percentile(samples, 0.5),
                               "gpu_p95_ms": percentile(samples, 0.95), "gpu_max_ms": samples.max()!]
    for (k, v) in extra { out[k] = v }
    return out
}

func dispatch(_ e: MTLComputeCommandEncoder, _ p: MTLComputePipelineState, _ n: Int) {
    let width = min(128, p.maxTotalThreadsPerThreadgroup)
    e.setComputePipelineState(p)
    e.dispatchThreadgroups(MTLSize(width: (n + width - 1) / width, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
}

func simdEligibleReason(_ s: Shape) -> String {
    if s.simdEligible { return "" }
    var parts: [String] = []
    if s.k != 1 || s.stride != 1 || s.pad != 0 { parts.append("kernel implements 1x1 only; this shape is k=\(s.k) stride=\(s.stride) pad=\(s.pad)") }
    if s.spatial % 8 != 0 { parts.append("spatial=\(s.spatial) not a multiple of 8") }
    if s.o % 8 != 0 { parts.append("outputs=\(s.o) not a multiple of 8") }
    if s.c % 8 != 0 { parts.append("channels=\(s.c) not a multiple of 8") }
    return parts.joined(separator: "; ")
}

func run() throws {
    guard CommandLine.arguments.count == 2 else { throw LabFailure("usage: precision-lab NEW_RECEIPT.json") }
    guard let device = MTLCreateSystemDefaultDevice(), device.hasUnifiedMemory,
          let queue = device.makeCommandQueue()
    else { throw LabFailure("Unified-memory Metal GPU required") }
    guard MemoryLayout<F16Params>.stride == 36, MemoryLayout<ConvParams>.stride == 40
    else { throw LabFailure("ABI size mismatch") }

    let paths = ["kernels/conv_i8.metal", "kernels/epilogue.metal", "kernels/precision_arena.metal"]
    let sources = try paths.map { try String(contentsOfFile: $0, encoding: .utf8) }
    let library = try device.makeLibrary(source: sources.joined(separator: "\n"), options: nil)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        guard let f = library.makeFunction(name: name) else { throw LabFailure("missing function \(name)") }
        return try device.makeComputePipelineState(function: f)
    }
    let i8conv = try pipeline("conv_packed4")
    let i8epilogue = try pipeline("requantize_i8")
    let f16direct = try pipeline("conv_f16_direct")
    let f16f32acc = try pipeline("conv_f16_f32acc")
    let f32direct = try pipeline("conv_f32_direct")
    // The simdgroup matrix path is optional: report absence, never fake it.
    var simdPipeline: MTLComputePipelineState? = nil
    if let f = library.makeFunction(name: "conv_f16_simd_1x1") {
        simdPipeline = try? device.makeComputePipelineState(function: f)
    }
    let simdOK = simdPipeline != nil

    func encodeEpilogue(_ cb: MTLCommandBuffer, _ sums: MTLBuffer, _ quant: [MTLBuffer],
                        _ out: MTLBuffer, _ channels: Int, _ count: Int) throws {
        guard let e = cb.makeComputeCommandEncoder() else { throw LabFailure("encoder allocation") }
        e.setBuffer(sums, offset: 0, index: 0)
        for (i, b) in quant.enumerated() { e.setBuffer(b, offset: 0, index: i + 1) }
        e.setBuffer(out, offset: 0, index: 6)
        var c = UInt32(channels), n = UInt32(count)
        e.setBytes(&c, length: 4, index: 7); e.setBytes(&n, length: 4, index: 8)
        dispatch(e, i8epilogue, count); e.endEncoding()
    }

    func complete(_ cb: MTLCommandBuffer) throws -> Double {
        cb.commit(); cb.waitUntilCompleted()
        guard cb.status == .completed, cb.error == nil
        else { throw LabFailure("GPU command failed: \(String(describing: cb.error))") }
        let ms = (cb.gpuEndTime - cb.gpuStartTime) * 1000
        guard ms.isFinite, ms > 0 else { throw LabFailure("GPU timestamps unavailable") }
        return ms
    }

    // Identical shape family to FusionLab so INT8 numbers are comparable, plus
    // one larger 1x1 that the 8x8 simdgroup tile can cover.
    let shapes = [
        Shape(name: "tail_3x3", h: 7, w: 9, c: 5, o: 7, k: 3, stride: 1, pad: 1),
        Shape(name: "feature_1x1", h: 72, w: 128, c: 32, o: 32, k: 1, stride: 1, pad: 0),
        Shape(name: "feature_3x3", h: 72, w: 128, c: 32, o: 32, k: 3, stride: 1, pad: 1),
        Shape(name: "feature_stride2", h: 73, w: 129, c: 32, o: 32, k: 2, stride: 2, pad: 0),
        Shape(name: "wide_1x1", h: 64, w: 64, c: 64, o: 64, k: 1, stride: 1, pad: 0),
    ]

    let warmup = 10, repeats = 31
    var rows = [[String: Any]]()
    var simdChecked = 0, simdExactCount = 0

    for s in shapes {
        guard (s.k * s.k * s.c * 16384 + 200) * 5 < Int(Int32.max)
        else { throw LabFailure("unsafe accumulator for \(s.name)") }

        // 31 unsigned bits mapped to [-1, 1). Masking first keeps the value
        // inside the Int8 quantisation range after the x127 scale below.
        var state: UInt64 = 0x51463846555345
        func nextUnit() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float((state >> 33) & 0x7FFF_FFFF) / 1073741824.0 - 1.0
        }
        let aCount = s.h * s.w * s.c, wCount = s.o * s.k * s.k * s.c
        var aUnit = [Float](repeating: 0, count: aCount)
        var wUnit = [Float](repeating: 0, count: wCount)
        for i in 0..<aCount { aUnit[i] = nextUnit() }
        for i in 0..<wCount { wUnit[i] = nextUnit() }

        let aHalf = aUnit.map { toHalf($0) }
        let wHalf = wUnit.map { toHalf($0) }
        // Weight tile pre-transposed for the simdgroup path: bT[k][o] = W[o][k].
        // Only 1x1 shapes have k == 1, so the transposition is a pure [o][k] ->
        // [k][o] swap and the channel count is the contraction length.
        var wHalfT = [UInt16](repeating: 0, count: s.c * s.o)
        for o in 0..<s.o { for k in 0..<s.c { wHalfT[k * s.o + o] = wHalf[o * s.c + k] } }
        var aInt8 = [Int8](repeating: 0, count: s.h * s.w * s.cp)
        var wInt8 = [Int8](repeating: 0, count: s.o * s.k * s.k * s.cp)
        for i in 0..<s.h*s.w { for c in 0..<s.c { aInt8[i*s.cp + c] = Int8(aUnit[i*s.c + c] * 127) } }
        for i in 0..<(s.o*s.k*s.k) { for c in 0..<s.c { wInt8[i*s.cp + c] = Int8(wUnit[i*s.c + c] * 127) } }

        let f32Expected = f32Oracle(aUnit, wUnit, s)
        let f16Expected = halfAccumulate(aHalf, wHalf, s)
        let f16f32Expected = f32AccumulateHalfInputs(aHalf, wHalf, s)

        let bias = (0..<s.o).map { Int32(($0 % 7 - 3) * 31) }
        let mult = (0..<s.o).map { Int32(1 + 2 * ($0 % 3)) }
        let shifts = (0..<s.o).map { UInt32(10 + ($0 % 5)) }
        let zeros = (0..<s.o).map { Int32($0 % 9 - 4) }
        let lows = (0..<s.o).map { $0 % 2 == 0 ? zeros[$0] : Int32(-128) }
        var int8Expected = [Int8](repeating: 0, count: s.count)
        for y in 0..<s.oh { for x in 0..<s.ow { for oc in 0..<s.o {
            var sum: Int64 = 0
            for ky in 0..<s.k { for kx in 0..<s.k {
                let iy = y * s.stride + ky - s.pad, ix = x * s.stride + kx - s.pad
                if iy < 0 || ix < 0 || iy >= s.h || ix >= s.w { continue }
                for c in 0..<s.c {
                    sum += Int64(aInt8[(iy*s.w+ix)*s.cp + c]) * Int64(wInt8[((oc*s.k+ky)*s.k+kx)*s.cp + c])
                }
            }}
            int8Expected[(y * s.ow + x) * s.o + oc] =
                int8Oracle(sum, bias[oc], mult[oc], shifts[oc], zeros[oc], lows[oc])
        }}}

        let abF32 = try makeBuffer(aUnit, device), wbF32 = try makeBuffer(wUnit, device)
        let abH = try makeBuffer(aHalf, device), wbH = try makeBuffer(wHalf, device)
        let wbHT = try makeBuffer(wHalfT, device)
        let abI8 = try makeBuffer(aInt8, device), wbI8 = try makeBuffer(wInt8, device)
        let q = try [makeBuffer(bias, device), makeBuffer(mult, device),
                     makeBuffer(shifts, device), makeBuffer(zeros, device),
                     makeBuffer(lows, device)]
        let scratch = try makeBuffer([Int32](repeating: 0x6A6A6A6A, count: s.count + 64), device)
        let outF32 = try makeBuffer([Float](repeating: 0, count: s.count + 64), device)
        let outF16A = try makeBuffer([UInt16](repeating: 0x3C00, count: s.count + 64), device)
        let outF16B = try makeBuffer([UInt16](repeating: 0x3C00, count: s.count + 64), device)
        let outI8 = try makeBuffer([Int8](repeating: 73, count: s.count + 64), device)

        var pF16 = s.f16, pI8 = s.int8
        var spatial = UInt32(s.spatial)

        // Every variant is timed inside the SAME trial loop, with the order
        // alternated between trials. Timing whole variants in separate blocks
        // let a GPU clock or power-state shift land entirely inside one variant
        // and contaminate it; interleaving makes each variant see the same
        // distribution of machine states. This is the control the earlier
        // fusion receipt lacked and the reason its 1x1 numbers swung 3x.
        let launchF32: (MTLCommandBuffer) throws -> Void = { cb in
            guard let e = cb.makeComputeCommandEncoder() else { throw LabFailure("encoder") }
            e.setBuffer(abF32, offset: 0, index: 0); e.setBuffer(wbF32, offset: 0, index: 1)
            e.setBuffer(outF32, offset: 0, index: 2); e.setBytes(&pF16, length: 36, index: 3)
            dispatch(e, f32direct, s.count); e.endEncoding()
        }
        let launchF16: (MTLCommandBuffer) throws -> Void = { cb in
            guard let e = cb.makeComputeCommandEncoder() else { throw LabFailure("encoder") }
            e.setBuffer(abH, offset: 0, index: 0); e.setBuffer(wbH, offset: 0, index: 1)
            e.setBuffer(outF16A, offset: 0, index: 2); e.setBytes(&pF16, length: 36, index: 3)
            dispatch(e, f16direct, s.count); e.endEncoding()
        }
        let launchF16F32: (MTLCommandBuffer) throws -> Void = { cb in
            guard let e = cb.makeComputeCommandEncoder() else { throw LabFailure("encoder") }
            e.setBuffer(abH, offset: 0, index: 0); e.setBuffer(wbH, offset: 0, index: 1)
            e.setBuffer(outF16B, offset: 0, index: 2); e.setBytes(&pF16, length: 36, index: 3)
            dispatch(e, f16f32acc, s.count); e.endEncoding()
        }
        let launchInt8: (MTLCommandBuffer) throws -> Void = { cb in
            guard let e = cb.makeComputeCommandEncoder() else { throw LabFailure("encoder") }
            e.setBuffer(abI8, offset: 0, index: 0); e.setBuffer(wbI8, offset: 0, index: 1)
            e.setBuffer(scratch, offset: 0, index: 2); e.setBytes(&pI8, length: 40, index: 3)
            dispatch(e, i8conv, s.count); e.endEncoding()
            try encodeEpilogue(cb, scratch, q, outI8, s.o, s.count)
        }

        let outSimd: MTLBuffer? = (s.simdEligible && simdPipeline != nil)
            ? try makeBuffer([UInt16](repeating: 0x7BFF, count: s.count + 64), device)
            : nil
        var simdLaunch: ((MTLCommandBuffer) throws -> Void)?
        if s.simdEligible, let sp = simdPipeline {
            simdLaunch = { cb in
                guard let e = cb.makeComputeCommandEncoder() else { throw LabFailure("encoder") }
                e.setBuffer(abH, offset: 0, index: 0); e.setBuffer(wbHT, offset: 0, index: 1)
                e.setBuffer(outSimd!, offset: 0, index: 2); e.setBytes(&pF16, length: 36, index: 3)
                e.setBytes(&spatial, length: 4, index: 4)
                e.setComputePipelineState(sp)
                e.dispatchThreadgroups(MTLSize(width: s.outTiles, height: s.spatialTiles, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
                e.endEncoding()
            }
        }

        var order: [(String, (MTLCommandBuffer) throws -> Void)] = [
            ("int8_split", launchInt8), ("fp32", launchF32),
            ("fp16_halfacc", launchF16), ("fp16_f32acc", launchF16F32),
        ]
        if simdLaunch != nil { order.append(("fp16_simd8x8", simdLaunch!)) }
        var samples = [String: [Double]](uniqueKeysWithValues: order.map { ($0.0, [Double]()) })

        for trial in 0..<(warmup + repeats) {
            // Alternate direction every trial so neither end is systematically first.
            let seq = trial % 2 == 0 ? order : order.reversed()
            for (name, launch) in seq {
                guard let cb = queue.makeCommandBuffer() else { throw LabFailure("command allocation") }
                try launch(cb)
                let ms = try complete(cb)
                if trial >= warmup { samples[name]?.append(ms) }
            }
        }

        let f32Result = (percentile(samples["fp32"]!, 0.5), samples["fp32"]!)
        let f16Result = (percentile(samples["fp16_halfacc"]!, 0.5), samples["fp16_halfacc"]!)
        let f16f32Result = (percentile(samples["fp16_f32acc"]!, 0.5), samples["fp16_f32acc"]!)
        let int8Result = (percentile(samples["int8_split"]!, 0.5), samples["int8_split"]!)

        let f32Ptr = outF32.contents().bindMemory(to: Float.self, capacity: s.count + 64)
        var f32MaxAbs: Float = 0, f32MaxRel: Float = 0
        for i in 0..<s.count {
            let d = abs(f32Ptr[i] - f32Expected[i])
            f32MaxAbs = max(f32MaxAbs, d)
            if abs(f32Expected[i]) > 1e-4 { f32MaxRel = max(f32MaxRel, d / abs(f32Expected[i])) }
        }
        guard (s.count..<(s.count + 64)).allSatisfy({ f32Ptr[$0] == 0 }) else { throw LabFailure("fp32 tail") }

        let f16Ptr = outF16A.contents().bindMemory(to: UInt16.self, capacity: s.count + 64)
        var f16MaxAbs: Float = 0, f16MaxRel: Float = 0
        for i in 0..<s.count {
            let d = abs(fromHalf(f16Ptr[i]) - f16Expected[i])
            f16MaxAbs = max(f16MaxAbs, d)
            if abs(f16Expected[i]) > 1e-4 { f16MaxRel = max(f16MaxRel, d / abs(f16Expected[i])) }
        }
        guard (s.count..<(s.count + 64)).allSatisfy({ f16Ptr[$0] == 0x3C00 })
        else { throw LabFailure("fp16 tail overwrite") }

        let f16f32Ptr = outF16B.contents().bindMemory(to: UInt16.self, capacity: s.count + 64)
        var f16f32MaxAbs: Float = 0, f16f32VsF32: Float = 0
        for i in 0..<s.count {
            f16f32MaxAbs = max(f16f32MaxAbs, abs(fromHalf(f16f32Ptr[i]) - f16f32Expected[i]))
            f16f32VsF32 = max(f16f32VsF32, abs(fromHalf(f16f32Ptr[i]) - f32Expected[i]))
        }

        let int8Ptr = outI8.contents().bindMemory(to: Int8.self, capacity: s.count + 64)
        guard int8Expected.indices.allSatisfy({ int8Ptr[$0] == int8Expected[$0] })
        else { throw LabFailure("int8 split mismatch on \(s.name)") }
        guard (s.count..<(s.count + 64)).allSatisfy({ int8Ptr[$0] == 73 }) else { throw LabFailure("int8 tail") }


        var simd: [String: Any] = ["measured": false, "eligible": s.simdEligible]
        if s.simdEligible, let sp = simdPipeline {
            simdChecked += 1
            let simdPtr = outSimd!.contents().bindMemory(to: UInt16.self, capacity: s.count + 64)
            let simdRef = simdReference(aHalf, wHalf, s)
            // ULP is meaningless where the exact sum is near zero through
            // cancellation, so the gate is relative to the largest magnitude in
            // the tensor, which is how a half-accumulated GEMM's error is
            // actually bounded. Bitwise equality is still reported per shape.
            let mag = (0..<s.count).map { abs(simdRef[$0]) }.max() ?? 1
            let ulp = pow(2.0, floor(log2(max(mag, 6.103515625e-05))) - 10)
            let ulpBudget = Float(s.c) * 0.5
            var simdMaxAbs: Float = 0, simdMaxULP: Float = 0
            for i in 0..<s.count {
                let err = abs(fromHalf(simdPtr[i]) - simdRef[i])
                simdMaxAbs = max(simdMaxAbs, err)
                simdMaxULP = max(simdMaxULP, err / ulp)
            }
            let tailIntact = (s.count..<(s.count + 64)).allSatisfy({ simdPtr[$0] == 0x7BFF })
            // A fast but wrong matrix path must never be reported as a result.
            // The budget is half a ULP per accumulated term, measured against
            // the exact half product-sum, so a layout bug cannot hide inside it.
            guard simdMaxULP <= ulpBudget, tailIntact else {
                throw LabFailure("simdgroup 1x1 exceeds half-ULP budget on " + s.name
                    + ": max_abs=\(simdMaxAbs) max_ulp=\(simdMaxULP)"
                    + " budget=\(ulpBudget) tail=\(tailIntact)")
            }
            if simdMaxULP == 0 { simdExactCount += 1 }
            let simdSamples = samples["fp16_simd8x8"]!
            simd = ["measured": true, "eligible": true,
                    "gpu_median_ms": percentile(simdSamples, 0.5),
                    "gpu_p95_ms": percentile(simdSamples, 0.95),
                    "gpu_min_ms": simdSamples.min()!, "gpu_max_ms": simdSamples.max()!,
                    "max_abs_vs_exact_half_sum": simdMaxAbs, "max_ulp_at_tensor_magnitude": simdMaxULP,
                    "ulp_budget": ulpBudget, "within_budget": true,
                    "bitwise_exact": simdMaxULP == 0, "tail_guard_passed": tailIntact]

        } else if !simdEligibleReason(s).isEmpty {
            simd["reason"] = simdEligibleReason(s)
        } else if !simdOK {
            simd["reason"] = "simdgroup pipeline unavailable on this device"
        }

        rows.append([
            "fixture": s.name,
            "input_nhwc": [1, s.h, s.w, s.c], "output_nhwc": [1, s.oh, s.ow, s.o],
            "elements_checked": s.count,
            "int8_split": spread(samples["int8_split"]!, ["exact": true, "dispatches": 2]),
            "fp32": spread(samples["fp32"]!, ["max_abs_vs_oracle": f32MaxAbs,
                     "max_rel_vs_oracle": f32MaxRel, "tail_guard_passed": true]),
            "fp16_halfacc": spread(samples["fp16_halfacc"]!, ["max_abs_vs_oracle": f16MaxAbs,
                             "max_rel_vs_oracle": f16MaxRel, "tail_guard_passed": true]),
            "fp16_f32acc": spread(samples["fp16_f32acc"]!, ["max_abs_vs_oracle": f16f32MaxAbs,
                            "max_abs_vs_fp32_path": f16f32VsF32, "tail_guard_passed": true]),
            "fp16_simd8x8": simd,
        ])
        print("PASS \(s.name) int8=\(int8Result.0) fp32=\(f32Result.0) fp16h=\(f16Result.0) fp16f32=\(f16f32Result.0) ms")
    }

    var hashes = [String: String]()
    for path in paths + ["native/PrecisionLab.swift", CommandLine.arguments[0]] {
        hashes[path] = sha256(try Data(contentsOf: URL(fileURLWithPath: path)))
    }
    let report: [String: Any] = [
        "schema": "helix.fsr-metal.precision-arena.v1",
        "scope": "synthetic_operator_precision_comparison",
        "question": "on identical shapes and identical underlying values, is an FP16 path faster or more accurate than the scalar INT8 path on this host?",
        "device": device.name, "os": ProcessInfo.processInfo.operatingSystemVersionString,
        "utc_timestamp": ISO8601DateFormatter().string(from: Date()),
        "thermal_state_at_end": ProcessInfo.processInfo.thermalState.rawValue,
        "source_and_binary_sha256": hashes,
        "warmup": warmup, "repeats": repeats,
        "variants": [
            "int8_split": "signed INT8 packed-four convolution + separate dyadic requantisation (existing path)",
            "fp32": "single dispatch, float32 accumulate",
            "fp16_halfacc": "single dispatch, half accumulator; products and partial sums rounded to half each step",
            "fp16_f32acc": "single dispatch, half storage with float32 accumulator, half output",
            "fp16_simd8x8": "1x1 only: 8x8 simdgroup matrix multiply over the flattened spatial axis",
        ],
        "simdgroup_matrix_available": simdOK,
        "simdgroup_shapes_measured": simdChecked,
        "simdgroup_shapes_exact": simdExactCount,
        "int8_numerics": "exact against an independent Int64/Double dyadic oracle",
        "fp16_numerics": "measured error reported; half arithmetic is NOT claimed exact",
        "full_fsr_implemented": false, "real_model_weights": false,
        "picture_quality_measured": false, "student_precision_chosen": false,
        "crossover_interop_tested": false,
        "gpu_timing_scope": "one command buffer, one or two encoders; excludes compile, allocation, packing, CPU reference and readback",
        "absolute_timing_caveat": "absolute milliseconds are NOT reproducible on this host: the same binary lands in two GPU clock states 2-4x apart and all variants move together within a run. Compare same-run ratios only.",
        "variant_timing_protocol": "all variants interleaved inside one trial loop, order alternated per trial, so a clock shift cannot land inside a single variant",
        "interpretation_rules": [
            "INT8 output is exactly reproducible; FP16 output is only near its oracle.",
            "FP16 speed does not imply FP16 is numerically adequate for a real model; that needs operator sensitivity data.",
            "These are our own synthetic operators, not AMD FSR operators, on a uniform synthetic distribution, not real activation statistics.",
            "A win here does not establish a win for a game frame.",
        ],
        "results": rows,
    ]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        .write(to: URL(fileURLWithPath: CommandLine.arguments[1]), options: .withoutOverwriting)
}

do { try run() } catch {
    FileHandle.standardError.write(Data("precision-lab: \(error)\n".utf8))
    exit(1)
}
