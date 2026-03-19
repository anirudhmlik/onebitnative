package ai.onebit.nativeapp

object OneBitNativeBridge {
  init {
    System.loadLibrary("onebit")
  }

  external fun ping(): String

  external fun initBackend()
  external fun loadModel(modelPath: String, nCtx: Int, nThreads: Int)
  external fun unloadModel()

  /**
   * Start generation on a native background thread.
   * Tokens stream back via the callback.
   */
  external fun generate(
    requestId: String,
    prompt: String,
    maxTokens: Int,
    temperature: Float,
    topP: Float,
    topK: Int,
    callback: OneBitCallback,
  )

  external fun stop()
}

interface OneBitCallback {
  fun onToken(requestId: String, tokenPiece: String)
  fun onDone(requestId: String, fullText: String, tokens: Int, tps: Float)
  fun onError(requestId: String, message: String)
}

