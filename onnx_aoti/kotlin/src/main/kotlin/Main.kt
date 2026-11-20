import java.nio.file.Path

fun main(args: Array<String>) {
    val onnxPath = Path.of(args.getOrElse(0) { "tests/artifacts/small_net.onnx" })
    val inputShape = args.getOrNull(1)?.split(",")?.map { it.trim().toLong() }?.toLongArray()

    val lib = AotiLibrary.default()

    val t0 = System.nanoTime()
    lib.loadOnnx(
        onnxPath = onnxPath,
        inputShape = inputShape,
        dtype = AotiDtype.FLOAT32,
        numRunners = 4,
    ).use { model ->
        println("load: ${(System.nanoTime() - t0) / 1_000_000} ms, input=${model.inputShape.toList()}, output=${model.outputShape.toList()}")

        val data = FloatArray(model.inputNumel.toInt()) { 0.1f }

        // predict(): one call, output allocated per call (includes allocation and one copy).
        FloatTensor.of(data, *model.inputShape).use { input ->
            repeat(20) { model.predict(input).close() }   // warm-up
            val runs = 500
            val t2 = System.nanoTime()
            repeat(runs) { model.predict(input).close() }
            val perCallUs = (System.nanoTime() - t2) / 1_000.0 / runs
            println("predict(): %.1f us per call over %d calls".format(perCallUs, runs))
        }

        model.newSession().use { session ->
            session.input.copyFrom(data)
            val t1 = System.nanoTime()
            repeat(1000) { session.run() }
            println("1000 runs: ${(System.nanoTime() - t1) / 1_000_000} ms")
            println("Output: ${session.output.toFloatArray().take(5)}")
        }
    }
}
