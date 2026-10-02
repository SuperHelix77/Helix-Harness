import Foundation
import Metal
import CryptoKit

struct Failure: Error, CustomStringConvertible {
    let description: String
    init(_ s: String) { description = s }
}
struct Params {
    var height, width, channels, outputs, kernelSize, stride, padding: UInt32
    var outHeight, outWidth, paddedChannels: UInt32
}
struct Shape {
    let name: String
    let h, w, c, o, k, stride, pad: Int
    var cp: Int { (c+3)/4*4 }
    var oh: Int { (h+2*pad-k)/stride+1 }
    var ow: Int { (w+2*pad-k)/stride+1 }
    var count: Int { oh*ow*o }
    var params: Params { Params(height: UInt32(h), width: UInt32(w), channels: UInt32(c),
        outputs: UInt32(o), kernelSize: UInt32(k), stride: UInt32(stride), padding: UInt32(pad),
        outHeight: UInt32(oh), outWidth: UInt32(ow), paddedChannels: UInt32(cp)) }
}
func checksum(_ data: Data) -> String { SHA256.hash(data: data).map { String(format:"%02x",$0) }.joined() }
func percentile(_ a: [Double], _ p: Double) -> Double { a.sorted()[max(0,Int(ceil(Double(a.count)*p))-1)] }
func buffer<T>(_ a: [T], _ d: MTLDevice) throws -> MTLBuffer {
    guard let b = a.withUnsafeBytes({ d.makeBuffer(bytes:$0.baseAddress!,length:$0.count,options:.storageModeShared) }) else {
        throw Failure("allocation failed")
    }
    return b
}
func dispatch(_ e: MTLComputeCommandEncoder, _ p: MTLComputePipelineState, _ n: Int) {
    let width = min(128,p.maxTotalThreadsPerThreadgroup)
    e.setComputePipelineState(p)
    e.dispatchThreadgroups(MTLSize(width:(n+width-1)/width,height:1,depth:1),
        threadsPerThreadgroup:MTLSize(width:width,height:1,depth:1))
}
func oracle(_ sum: Int64, _ b: Int32, _ m: Int32, _ shift: UInt32, _ z: Int32, _ lower: Int32) -> Int8 {
    // Independent Double oracle: bounded integer products and powers of two
    // are represented exactly. GPU uses integer quotient/remainder logic.
    let value = (Double((sum+Int64(b))*Int64(m))/Double(UInt64(1)<<shift)).rounded(.toNearestOrEven)
    return Int8(max(Int64(lower),min(127,Int64(value)+Int64(z))))
}
func run() throws {
    guard CommandLine.arguments.count == 2 else { throw Failure("usage: fusion-lab NEW_RECEIPT.json") }
    guard let device = MTLCreateSystemDefaultDevice(), device.hasUnifiedMemory,
          let queue = device.makeCommandQueue() else { throw Failure("Unified-memory Metal GPU required") }
    guard MemoryLayout<Params>.stride == 40 else { throw Failure("ABI size mismatch") }
    let paths = ["kernels/conv_i8.metal","kernels/epilogue.metal"]
    let sources = try paths.map { try String(contentsOfFile:$0,encoding:.utf8) }
    let library = try device.makeLibrary(source:sources.joined(separator:"\n"),options:nil)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        guard let f = library.makeFunction(name:name) else { throw Failure("missing function \(name)") }
        return try device.makeComputePipelineState(function:f)
    }
    let conv = try pipeline("conv_packed4"), epilogue = try pipeline("requantize_i8"), fused = try pipeline("conv_fused_i8")
    func encodeEpilogue(_ cb: MTLCommandBuffer, _ sums: MTLBuffer, _ quant: [MTLBuffer], _ out: MTLBuffer, _ channels: Int, _ count: Int) throws {
        guard let e = cb.makeComputeCommandEncoder() else { throw Failure("encoder allocation") }
        e.setBuffer(sums,offset:0,index:0)
        for (i,b) in quant.enumerated() { e.setBuffer(b,offset:0,index:i+1) }
        e.setBuffer(out,offset:0,index:6)
        var c = UInt32(channels), n = UInt32(count)
        e.setBytes(&c,length:4,index:7); e.setBytes(&n,length:4,index:8)
        dispatch(e,epilogue,count); e.endEncoding()
    }
    func complete(_ cb: MTLCommandBuffer) throws -> Double {
        cb.commit(); cb.waitUntilCompleted()
        guard cb.status == .completed, cb.error == nil else { throw Failure("GPU command failed: \(String(describing:cb.error))") }
        let ms = (cb.gpuEndTime-cb.gpuStartTime)*1000
        guard ms.isFinite && ms > 0 else { throw Failure("GPU timestamps unavailable") }
        return ms
    }
    // Test every signed half-integer near zero plus saturation, channel bias,
    // per-channel scales/zero points and quantized ReLU on actual Metal.
    let roundInputs: [Int32] = Array(-1031...1031)
    let rb: [Int32] = [0,1,-3,7], rm: [Int32] = [1,3,5,1], rs: [UInt32] = [1,2,3,4]
    let rz: [Int32] = [0,-7,4,0], rl: [Int32] = [-128,-128,4,0]
    let quantRound = try [buffer(rb,device),buffer(rm,device),buffer(rs,device),buffer(rz,device),buffer(rl,device)]
    let roundOut = try buffer([Int8](repeating:73,count:roundInputs.count+64),device)
    guard let roundCommand = queue.makeCommandBuffer() else { throw Failure("command allocation") }
    try encodeEpilogue(roundCommand,buffer(roundInputs,device),quantRound,roundOut,4,roundInputs.count)
    _ = try complete(roundCommand)
    let rounded = roundOut.contents().bindMemory(to:Int8.self,capacity:roundInputs.count+64)
    for i in roundInputs.indices {
        let c = i%4
        guard rounded[i] == oracle(Int64(roundInputs[i]),rb[c],rm[c],rs[c],rz[c],rl[c]) else { throw Failure("rounding oracle mismatch") }
    }
    guard (roundInputs.count..<(roundInputs.count+64)).allSatisfy({rounded[$0] == 73}) else { throw Failure("rounding tail overwrite") }
    // Independent scalar tie examples, not only CPU/GPU agreement.
    for (v,e) in zip([-7,-5,-3,-1,1,3,5,7],[-4,-2,-2,0,0,2,2,4]) {
        guard oracle(Int64(v),0,1,1,0,-128) == Int8(e) else { throw Failure("CPU tie sanity") }
    }
    let shapes = [
        Shape(name:"tail_3x3",h:7,w:9,c:5,o:7,k:3,stride:1,pad:1),
        Shape(name:"feature_1x1",h:72,w:128,c:32,o:32,k:1,stride:1,pad:0),
        Shape(name:"feature_3x3",h:72,w:128,c:32,o:32,k:3,stride:1,pad:1),
        Shape(name:"feature_stride2",h:73,w:129,c:32,o:32,k:2,stride:2,pad:0)]
    var rows = [[String:Any]]()
    let warmup = 10, repeats = 31
    for s in shapes {
        guard (s.k*s.k*s.c*16384+200)*5 < Int(Int32.max) else { throw Failure("unsafe accumulator") }
        var state: UInt64 = 0x51463846555345
        func next() -> Int8 { state = state &* 6364136223846793005 &+ 1442695040888963407; return Int8(truncatingIfNeeded:state>>32) }
        var a = [Int8](repeating:0,count:s.h*s.w*s.cp), w = [Int8](repeating:0,count:s.o*s.k*s.k*s.cp)
        for i in 0..<(s.h*s.w) { for c in 0..<s.c { a[i*s.cp+c] = next() } }
        for i in 0..<(s.o*s.k*s.k) { for c in 0..<s.c { w[i*s.cp+c] = next() } }
        let b = (0..<s.o).map { Int32(($0%7-3)*31) }, m = (0..<s.o).map { Int32(1+2*($0%3)) }
        let shifts = (0..<s.o).map { UInt32(10+($0%5)) }, zeros = (0..<s.o).map { Int32($0%9-4) }
        let lows = (0..<s.o).map { $0%2 == 0 ? zeros[$0] : Int32(-128) }
        var expected = [Int8](repeating:0,count:s.count)
        for y in 0..<s.oh { for x in 0..<s.ow { for oc in 0..<s.o {
            var sum: Int64 = 0
            for ky in 0..<s.k { for kx in 0..<s.k {
                let iy = y*s.stride+ky-s.pad, ix = x*s.stride+kx-s.pad
                if iy < 0 || ix < 0 || iy >= s.h || ix >= s.w { continue }
                for c in 0..<s.c { sum += Int64(a[(iy*s.w+ix)*s.cp+c])*Int64(w[((oc*s.k+ky)*s.k+kx)*s.cp+c]) }
            }}
            expected[(y*s.ow+x)*s.o+oc] = oracle(sum,b[oc],m[oc],shifts[oc],zeros[oc],lows[oc])
        }}}
        let ab = try buffer(a,device), wb = try buffer(w,device)
        let q = try [buffer(b,device),buffer(m,device),buffer(shifts,device),buffer(zeros,device),buffer(lows,device)]
        let scratch = try buffer([Int32](repeating:0x6A6A6A6A,count:s.count+64),device)
        let outputs = try [buffer([Int8](repeating:73,count:s.count+64),device),buffer([Int8](repeating:73,count:s.count+64),device)]
        var gpu = [[Double](),[Double]()], wall = [[Double](),[Double]()]
        for trial in 0..<(warmup+repeats) {
            for variant in (trial%2 == 0 ? [0,1] : [1,0]) {
                let begin = ProcessInfo.processInfo.systemUptime
                guard let cb = queue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else { throw Failure("command allocation") }
                var p = s.params
                e.setBuffer(ab,offset:0,index:0); e.setBuffer(wb,offset:0,index:1)
                e.setBuffer(variant == 0 ? scratch : outputs[variant],offset:0,index:2)
                e.setBytes(&p,length:40,index:3)
                if variant == 1 { for (i,b) in q.enumerated() { e.setBuffer(b,offset:0,index:i+4) } }
                dispatch(e,variant == 0 ? conv : fused,s.count); e.endEncoding()
                if variant == 0 { try encodeEpilogue(cb,scratch,q,outputs[variant],s.o,s.count) }
                let ms = try complete(cb)
                let elapsed = (ProcessInfo.processInfo.systemUptime-begin)*1000
                if trial >= warmup { gpu[variant].append(ms); wall[variant].append(elapsed) }
            }
        }
        let scratchPointer = scratch.contents().bindMemory(to:Int32.self,capacity:s.count+64)
        guard (s.count..<(s.count+64)).allSatisfy({scratchPointer[$0] == 0x6A6A6A6A}) else { throw Failure("scratch overrun") }
        for variant in 0..<2 {
            let ptr = outputs[variant].contents().bindMemory(to:Int8.self,capacity:s.count+64)
            guard expected.indices.allSatisfy({ptr[$0] == expected[$0]}),
                  (s.count..<(s.count+64)).allSatisfy({ptr[$0] == 73}) else { throw Failure("fusion output/tail mismatch") }
            rows.append(["fixture":s.name,"variant":variant == 0 ? "split" : "fused",
                "input_nhwc":[1,s.h,s.w,s.c],"output_nhwc":[1,s.oh,s.ow,s.o],"elements_checked":s.count,
                "exact":true,"tail_guards_passed":true,"gpu_samples_ms":gpu[variant],"wall_samples_ms":wall[variant],
                "gpu_median_ms":percentile(gpu[variant],0.5),"gpu_p95_ms":percentile(gpu[variant],0.95),
                "wall_median_ms":percentile(wall[variant],0.5),"wall_p95_ms":percentile(wall[variant],0.95),
                "dispatches":variant == 0 ? 2 : 1,"logical_intermediate_bytes":variant == 0 ? s.count*4 : 0,
                "output_sha256":checksum(Data(bytes:ptr,count:s.count))])
            print("PASS \(s.name) \(variant == 0 ? "split" : "fused") \(percentile(gpu[variant],0.5)) ms; exact")
        }
    }
    var hashes = [String:String]()
    for path in paths+["native/FusionLab.swift",CommandLine.arguments[0]] { hashes[path] = checksum(try Data(contentsOf:URL(fileURLWithPath:path))) }
    let report: [String:Any] = ["schema":"helix.fsr-metal.fusion-lab.v1","scope":"synthetic_operator_pipeline",
        "device":device.name,"os":ProcessInfo.processInfo.operatingSystemVersionString,
        "utc_timestamp":ISO8601DateFormatter().string(from:Date()),"thermal_state_at_end":ProcessInfo.processInfo.thermalState.rawValue,
        "source_and_binary_sha256":hashes,"warmup":warmup,"repeats":repeats,"variant_order":"alternating",
        "quantization":"dyadic per-channel, integer RNE, signed INT8 saturate, optional quantized ReLU; NOT AMD-verified",
        "rounding_cases":roundInputs.count,"rounding_exact":true,"full_fsr_implemented":false,
        "real_model_weights":false,"picture_quality_measured":false,"crossover_interop_tested":false,
        "gpu_timing_scope":"one command buffer, one or two encoders; excludes compile, allocation, packing, CPU reference and readback",
        "wall_timing_scope":"command construction through GPU completion; not full game frame time",
        "memory_scope":"logical per-variant requirement only; allocations for both variants coexist in this benchmark",
        "results":rows]
    try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys])
        .write(to:URL(fileURLWithPath:CommandLine.arguments[1]),options:.withoutOverwriting)
}
do { try run() } catch { FileHandle.standardError.write(Data("fusion-lab: \(error)\n".utf8)); exit(1) }
