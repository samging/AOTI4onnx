import java.io.ByteArrayOutputStream
import java.io.PrintStream
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.atomic.AtomicInteger
import kotlin.concurrent.thread

/**
 * Checks the cache: concurrent cold loads compile once, warm loads skip Python and
 * the hashing, edits invalidate the key, and a broken ONNX file reports the compiler error.
 * Usage: CacheCheckKt <model.onnx> [inputShape]
 */
fun main(args: Array<String>) {
    val onnx = Path.of(args[0])
    val shape = args.getOrNull(1)?.takeUnless { it.startsWith("--") }
        ?.split(",")?.map { it.trim().toLong() }?.toLongArray()
        ?: longArrayOf(1, 3, 8, 8)
    val cache = Files.createTempDirectory("onnx_aoti_cache_check")
    val compiler = OnnxCache.defaultCompilerScript()

    if ("--hash-only" in args) {
        val first = timed { OnnxCache.contentHash(onnx, cache) }
        val warm = timed { OnnxCache.contentHash(onnx, cache) }
        println("hash first call: ${first.first / 1000} ms, index hit: ${warm.first} µs")
        return
    }
    val lib = AotiLibrary.default()

    val captured = ByteArrayOutputStream()
    val originalErr = System.err
    System.setErr(PrintStream(captured, true))
    val failures = AtomicInteger(0)
    val t0 = System.nanoTime()
    try {
        (1..4).map {
            thread {
                try {
                    lib.loadOnnx(onnx, shape, cacheDir = cache).close()
                } catch (e: Exception) {
                    failures.incrementAndGet()
                }
            }
        }.forEach { it.join() }
    } finally {
        System.setErr(originalErr)
    }
    val coldMs = (System.nanoTime() - t0) / 1_000_000
    val compiles = captured.toString().lines().count { "compiling" in it }
    println("cold, 4 threads: $compiles compile(s), $failures failure(s), $coldMs ms")

    val t1 = System.nanoTime()
    lib.loadOnnx(onnx, shape, cacheDir = cache).close()
    println("warm load: ${(System.nanoTime() - t1) / 1_000_000} ms")

    // Hashing with the index: first call reads the file, later calls only check size and mtime.
    val hashFirst = timed { OnnxCache.contentHash(onnx, cache) }
    val hashWarm = timed { OnnxCache.contentHash(onnx, cache) }
    println("content hash: first ${hashFirst.first / 1000} ms, index hit ${hashWarm.first} µs (same hash: ${hashFirst.second == hashWarm.second})")

    // Editing the file (same size, new mtime) must produce a new key.
    val copy = Files.createTempFile("edit_check", ".onnx")
    Files.copy(onnx, copy, java.nio.file.StandardCopyOption.REPLACE_EXISTING)
    val before = OnnxCache.cacheKey(copy, shape, AotiDtype.FLOAT32, compiler, cache)
    val bytes = Files.readAllBytes(copy)
    bytes[bytes.size / 2] = (bytes[bytes.size / 2] + 1).toByte()
    Files.write(copy, bytes)
    Files.setLastModifiedTime(copy, java.nio.file.attribute.FileTime.fromMillis(System.currentTimeMillis() + 5_000))
    val after = OnnxCache.cacheKey(copy, shape, AotiDtype.FLOAT32, compiler, cache)
    println("edited file gets a new key: ${before != after}")
    Files.deleteIfExists(copy)

    val keyAuto = OnnxCache.cacheKey(onnx, null, AotiDtype.FLOAT32, compiler, cache)
    val keyFp16 = OnnxCache.cacheKey(onnx, shape, AotiDtype.FLOAT16, compiler, cache)
    val keyShape = OnnxCache.cacheKey(onnx, shape, AotiDtype.FLOAT32, compiler, cache)
    println("keys differ by shape/dtype: ${keyAuto != keyShape && keyShape != keyFp16 && keyAuto != keyFp16}")

    val bad = Files.createTempFile("broken", ".onnx")
    Files.writeString(bad, "not an onnx file")
    try {
        lib.loadOnnx(bad, shape, cacheDir = cache)
        println("broken model: NO ERROR (unexpected)")
    } catch (e: Exception) {
        println("broken model error: ${e.message?.lines()?.take(2)?.joinToString(" | ")}")
    }
}

/** Returns elapsed microseconds and the block's result. */
private fun <T> timed(block: () -> T): Pair<Long, T> {
    val t = System.nanoTime()
    val value = block()
    return Pair((System.nanoTime() - t) / 1_000, value)
}
