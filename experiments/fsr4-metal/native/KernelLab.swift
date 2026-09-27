import Foundation
import Metal
import CryptoKit

struct LabError: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}
struct ConvParams {
    var height, width, channels, outputs, kernel, stride, padding: UInt32
    var outHeight, outWidth, paddedChannels: UInt32
}
struct Shape {
    let name: String
    let h, w, c, o, k, stride, pad: Int
    let extrema: Bool
    init(_ name: String, _ h: Int, _ w: Int, _ c: Int, _ o: Int,
         _ k: Int, _ stride: Int, _ pad: Int, extrema: Bool = false) {
        self.name = name; self.h = h; self.w = w; self.c = c; self.o = o
        self.k = k; self.stride = stride; self.pad = pad; self.extrema = extrema
    }
    var cp: Int { (c + 3) / 4 * 4 }
    var oh: Int { (h + 2 * pad - k) / stride + 1 }
    var ow: Int { (w + 2 * pad - k) / stride + 1 }
    var count: Int { oh * ow * o }
    var params: ConvParams {
        ConvParams(height: UInt32(h), width: UInt32(w), channels: UInt32(c),
            outputs: UInt32(o), kernel: UInt32(k), stride: UInt32(stride),
            padding: UInt32(pad), outHeight: UInt32(oh), outWidth: UInt32(ow),
            paddedChannels: UInt32(cp))
    }
}
struct Generator {
    var state: UInt64 = 0x4653524D4554414C
    mutating func next() -> Int8 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Int8(truncatingIfNeeded: state >> 32)
    }
}
func fixture(_ s: Shape) -> ([Int8], [Int8]) {
    var g = Generator()
    let extremes: [Int8] = [-128, 127, -1, 0]
    func values(_ pixels: Int, _ generator: inout Generator) -> [Int8] {
        var result = [Int8](repeating: 0, count: pixels * s.cp)
        for i in 0..<pixels { for c in 0..<s.c {
            result[i*s.cp+c] = s.extrema ? extremes[(i+c) % 4] : generator.next()
        }}
        return result
    }
    return (values(s.h*s.w, &g), values(s.o*s.k*s.k, &g))
}
func reference(_ s: Shape, _ input: [Int8], _ weights: [Int8]) -> [Int32] {
    var result = [Int32](repeating: 0, count: s.count)
    for y in 0..<s.oh { for x in 0..<s.ow { for oc in 0..<s.o {
        var sum: Int64 = 0
        for ky in 0..<s.k { for kx in 0..<s.k {
            let iy = y*s.stride+ky-s.pad, ix = x*s.stride+kx-s.pad
            if iy < 0 || ix < 0 || iy >= s.h || ix >= s.w { continue }
            let a = (iy*s.w+ix)*s.cp, b = ((oc*s.k+ky)*s.k+kx)*s.cp
            for c in 0..<s.c { sum += Int64(input[a+c])*Int64(weights[b+c]) }
        }}
        result[(y*s.ow+x)*s.o+oc] = Int32(sum)
    }}}
    return result
}
func quantile(_ values: [Double], _ q: Double) -> Double {
    let sorted = values.sorted()
    return sorted[max(0, min(sorted.count-1, Int(ceil(Double(sorted.count)*q))-1))]
}
func sha(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

func main() throws {
    guard CommandLine.arguments.count == 3 else {
        throw LabError("Usage: kernel-lab kernels/conv_i8.metal results.json")
    }
    let sourceURL = URL(fileURLWithPath: CommandLine.arguments[1])
    let sourceData = try Data(contentsOf: sourceURL)
    guard let source = String(data: sourceData, encoding: .utf8),
          let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw LabError("UTF-8 kernel source and a Metal GPU are required; no CPU fallback")
    }
    guard device.hasUnifiedMemory else { throw LabError("This lab currently requires unified memory") }
    guard MemoryLayout<ConvParams>.stride == 40 else { throw LabError("Swift/Metal parameter ABI mismatch") }
    let library = try device.makeLibrary(source: source, options: nil)
    let names = ["conv_scalar", "conv_packed4"]
    let pipelines = try names.map { name -> MTLComputePipelineState in
        guard let function = library.makeFunction(name: name) else { throw LabError("Missing kernel: \(name)") }
        return try device.makeComputePipelineState(function: function)
    }
    let shapes = [
        Shape("signed_tail_1x1", 5, 7, 3, 5, 1, 1, 0),
        Shape("padding_tail_3x3", 7, 9, 5, 7, 3, 1, 1),
        Shape("stride2_odd_extent", 9, 11, 8, 8, 2, 2, 0),
        Shape("signed_extrema", 2, 2, 4, 4, 1, 1, 0, extrema: true),
        Shape("feature_1x1", 72, 128, 32, 32, 1, 1, 0),
        Shape("feature_3x3", 72, 128, 32, 32, 3, 1, 1),
        Shape("feature_downsample", 72, 128, 32, 32, 2, 2, 0)
    ]
    let warmup = 3, repeats = 21, guardCount = 64
    let sentinel: Int32 = 0x6A6A6A6A
    var rows = [[String: Any]]()
    for s in shapes {
        guard s.k*s.k*s.c*16384 <= Int(Int32.max), s.oh > 0, s.ow > 0 else {
            throw LabError("Invalid shape or accumulator overflow risk")
        }
        let (input, weights) = fixture(s)
        let expected = reference(s, input, weights)
        if s.extrema && expected.first != 32514 {
            throw LabError("CPU oracle failed the independently calculated signed-extrema check")
        }
        let a = input.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
        let b = weights.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared) }
        guard let a, let b else { throw LabError("Input allocation failed") }
        let outputs = try names.map { _ -> MTLBuffer in
            guard let buffer = device.makeBuffer(length: (s.count+guardCount)*4, options: .storageModeShared) else {
                throw LabError("Output allocation failed")
            }
            buffer.contents().bindMemory(to: Int32.self, capacity: s.count+guardCount)
                .initialize(repeating: sentinel, count: s.count+guardCount)
            return buffer
        }
        var timings = [[Double](), [Double]()]
        // Interleave variants to reduce fixed-order bias. No copies in GPU interval.
        for trial in 0..<(warmup+repeats) {
            let order = trial % 2 == 0 ? [0, 1] : [1, 0]
            for variant in order {
                guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
                    throw LabError("Command allocation failed")
                }
                let pipeline = pipelines[variant]
                var params = s.params
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(a, offset: 0, index: 0)
                encoder.setBuffer(b, offset: 0, index: 1)
                encoder.setBuffer(outputs[variant], offset: 0, index: 2)
                encoder.setBytes(&params, length: MemoryLayout<ConvParams>.stride, index: 3)
                let width = min(128, pipeline.maxTotalThreadsPerThreadgroup)
                // Round up deliberately: the kernel's tail guard must be correct.
                encoder.dispatchThreadgroups(MTLSize(width: (s.count+width-1)/width, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
                encoder.endEncoding()
                command.commit(); command.waitUntilCompleted()
                guard command.status == .completed, command.error == nil else {
                    throw LabError("GPU failure: \(String(describing: command.error))")
                }
                let ms = (command.gpuEndTime-command.gpuStartTime)*1000
                guard ms.isFinite && ms > 0 else { throw LabError("GPU timestamps unavailable") }
                if trial >= warmup { timings[variant].append(ms) }
            }
        }
        for variant in 0..<2 {
            let actual = outputs[variant].contents().bindMemory(to: Int32.self, capacity: s.count+guardCount)
            var maxError: Int64 = 0
            for i in 0..<s.count { maxError = max(maxError, abs(Int64(actual[i])-Int64(expected[i]))) }
            let guardOK = (s.count..<(s.count+guardCount)).allSatisfy { actual[$0] == sentinel }
            guard maxError == 0 && guardOK else {
                throw LabError("FAIL \(s.name)/\(names[variant]): error=\(maxError), guard=\(guardOK)")
            }
            let row: [String: Any] = ["fixture": s.name, "kernel": names[variant],
                "input_nhwc": [1,s.h,s.w,s.c], "output_nhwc": [1,s.oh,s.ow,s.o],
                "kernel_size": s.k, "stride": s.stride, "padding": s.pad,
                "padded_channels": s.cp, "elements_checked": s.count,
                "max_abs_error": maxError, "tail_guard_passed": guardOK,
                "gpu_median_ms": quantile(timings[variant],0.5),
                "gpu_p95_ms": quantile(timings[variant],0.95), "gpu_samples_ms": timings[variant],
                "output_sha256": sha(Data(bytes: actual, count: s.count*4))]
            rows.append(row)
            print("PASS \(s.name)/\(names[variant]) exact; GPU median \(quantile(timings[variant],0.5)) ms")
        }
    }
    let result: [String: Any] = ["schema": "helix.fsr-metal.kernel-lab.v1",
        "scope": "synthetic_int8_convolution_microbenchmark", "full_fsr_implemented": false,
        "crossover_interop_tested": false, "model_weights_loaded": false,
        "device": device.name, "registry_id": String(device.registryID),
        "unified_memory": device.hasUnifiedMemory,
        "os": ProcessInfo.processInfo.operatingSystemVersionString,
        "thermal_state_at_end": ProcessInfo.processInfo.thermalState.rawValue,
        "utc_timestamp": ISO8601DateFormatter().string(from: Date()),
        "kernel_source_sha256": sha(sourceData), "warmup_dispatches": warmup,
        "measured_dispatches": repeats, "variant_order": "alternating",
        "timing_scope": "one command buffer / one compute dispatch; excludes CPU reference, compilation, packing, allocation and readback; includes GPU command overhead",
        "accumulation": "signed int8 multiply, int32 accumulate; no bias/requantization/activation",
        "limitations": ["synthetic shapes, not a recovered FSR graph", "no FSR quality or end-to-end frame-time claim", "no isolated-machine or sustained-thermal guarantee"],
        "results": rows]
    try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
        .write(to: URL(fileURLWithPath: CommandLine.arguments[2]), options: .atomic)
}
do { try main() } catch {
    FileHandle.standardError.write(Data("kernel-lab: \(error)\n".utf8)); exit(1)
}
