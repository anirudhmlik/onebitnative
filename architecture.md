# OneBitNative — Architecture

## 1. Project Overview

OneBitNative is a single-screen Android chat application that runs an on-device **BitNet** (1-bit quantized) LLM for inference. The UI is built in **Kotlin + Jetpack Compose**, and the inference engine is a **C++ native library** (`libonebit.so`) linked via **JNI** against a BitNet-optimized fork of [llama.cpp](https://github.com/ggml-org/llama.cpp). The app targets **ARM64-v8a** only, uses **GGUF**-format model files, and has no network calls — all inference runs locally.

---

## 2. Domain Models

### 2.1 Kotlin Models

| Model | File | Fields | Purpose |
|---|---|---|---|
| `ChatMsg` | `MainActivity.kt:87` | `id: String`, `role: String`, `text: String`, `isError: Boolean` | Data class representing a single chat bubble. `role` is `"user"` or `"assistant"`. |
| `OneBitCallback` | `OneBitNativeBridge.kt:31` | `onToken(id, piece)`, `onDone(id, fullText, tokens, tps)`, `onError(id, msg)` | Interface that the native layer calls to stream results back to Kotlin. |

### 2.2 Implicit Parameter Models (passed to `generate`)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `requestId` | `String` | UUID | Correlates callbacks to the originating request. |
| `prompt` | `String` | user input | Raw user text (applied through chat template in native). |
| `maxTokens` | `Int` | `512` | Generation limit. |
| `temperature` | `Float` | `0.8` | Sampling temperature. |
| `topP` | `Float` | `0.95` | Nucleus sampling threshold. |
| `topK` | `Int` | `40` | Top-K sampling. |

### 2.3 Native C++ Models (globals, mutex-guarded)

| Variable | Type | Purpose |
|---|---|---|
| `g_model` | `llama_model*` | The loaded GGUF model. |
| `g_ctx` | `llama_context*` | Inference context (KV cache, threads). |
| `g_smpl` | `llama_sampler*` | Global sampler (allocated per request in practice). |
| `g_cancel` | `std::atomic<bool>` | Cancellation flag checked each iteration of the generation loop. |
| `g_mu` | `std::mutex` | Protects model/context/sampler from concurrent load/unload. |
| `g_cbClass` | `jclass` (global ref) | Cached JNI reference to the `OneBitCallback` class. |
| `g_onToken` / `g_onDone` / `g_onError` | `jmethodID` | Cached JNI method IDs for the callback interface. |

---

## 3. Services

There is no DI framework, ViewModel, or Repository pattern. The app has **one service facade** and **three native service modules**.

### 3.1 Bridge Service (`OneBitNativeBridge`)

**File:** `OneBitNativeBridge.kt`

A Kotlin `object` (singleton). Its `init` block calls `System.loadLibrary("onebit")` once. All methods are `external` and map to JNI functions in `onebit_jni.cpp`.

| Method | JNI Target | Behavior |
|---|---|---|
| `ping()` | `Java_…_ping` | Returns `"pong"` — connectivity test. |
| `initBackend()` | `Java_…_initBackend` | Calls `llama_backend_init()` under the global mutex. |
| `loadModel(path, nCtx, nThreads)` | `Java_…_loadModel` | Frees any existing model/context, loads a new GGUF file, creates context with given params. Throws `RuntimeException` on failure. |
| `unloadModel()` | `Java_…_unloadModel` | Frees sampler, context, and model under the mutex. |
| `generate(reqId, prompt, maxTokens, temp, topP, topK, cb)` | `Java_…_generate` | Spawns a detached native thread that runs the full generation pipeline. |
| `stop()` | `Java_…_stop` | Sets `g_cancel.store(true)` — causes the generation loop to exit on its next iteration. |

### 3.2 Native Model Manager

**File:** `onebit_jni.cpp` (globals)

Manages the lifecycle of `llama_model*` and `llama_context*`. Access is serialized by `g_mu`. Key parameters:
- `use_mmap = true` (memory-map the model file)
- `flash_attn = false` (CPU fallback is too slow)
- `type_k = type_v = GGML_TYPE_F16` (16-bit KV cache to avoid on-the-fly dequantization)
- `n_ctx = 2048`, `n_threads = n_threads_batch = 8`

### 3.3 Native Sampler Factory

**File:** `onebit_jni.cpp` (inside `generate`)

Creates a **per-request** sampler chain:
```
top_k(40) → top_p(0.95) → temperature(0.8) → distribution(seed)
```

### 3.4 Native Generation Engine

**File:** `onebit_jni.cpp:302-421`

A detached `std::thread` per `generate()` call. Workflow:
1. Attaches the JNI environment for the thread.
2. Copies model/context pointers under the mutex.
3. Creates the sampler chain.
4. Applies the **chat template** via `llama_chat_apply_template` with a single chat message `{"user", prompt}` and `add_ass=true` (prepends assistant-start tokens).
5. Tokenizes the full templated prompt.
6. **Full-prompt batch decode**: single `llama_decode` call with all tokens; falls back to last-token-only decode if the batch fails.
7. **Autoregressive loop**: sample → accept → check EOS/EOT → token-to-piece → stop-marker detection → `cb_token` → decode next.
8. **Stop-marker detection**: searches for markers like `User:`, `Assistant:`, `<|user|>`, `<|assistant|>`, `<|endofturn|>`, etc. in the accumulated output, trims before the first match.
9. **Final safety trim**: scans the full output again and truncates at the earliest stop marker.
10. Calculates TPS, fires `cb_done`.

---

## 4. Features

### 4.1 Model Import

| Step | Where | Detail |
|---|---|---|
| Trigger | `TopAppBar` menu → "Import .gguf Model" | `ActivityResultContracts.OpenDocument` with `*/*` MIME filter. |
| Copy | `MainActivity.importModelAndLoad()` | Copies picked file to `{filesDir}/models/ggml-model-i2_s.gguf` on `Dispatchers.IO`. |
| Load | Same method | After copy, calls `initBackend()` + `loadModel(path, 2048, 8)` on `Dispatchers.Default`. |

### 4.2 Auto-load on Startup

| Step | Where | Detail |
|---|---|---|
| Trigger | `OneBitRoot` → `LaunchedEffect(Unit)` | Fires once on first composition. |
| Check | `existingModelPath(ctx)` | Returns the absolute path if the file exists and is larger than 1 GB. |
| Load | Same `LaunchedEffect` | Calls `initBackend()` + `loadModel()` on `Dispatchers.Default`. |

### 4.3 Model Load/Unload Toggle

| Step | Where | Detail |
|---|---|---|
| Trigger | Top-app-bar `Clear` icon button | Visible only when `modelPath != null`. |
| Toggle logic | `scope.launch` in button `onClick` | Unloads if loaded; loads if unloaded. |
| Visual | Icon tint | Green when loaded, red when unloaded. |

### 4.4 Chat with Streaming Tokens

| Step | Where | Detail |
|---|---|---|
| Input | `BasicTextField` at bottom of `Scaffold` | Disabled when generating or no model. |
| Send | Circular button (changes to stop icon when generating) | Validates non-blank input and model loaded. |
| Message creation | Send button `onClick` | Appends user `ChatMsg` + empty assistant `ChatMsg` (with `requestId` as its `id`). |
| Native call | `OneBitNativeBridge.generate()` | Passes `OneBitCallback` instance. |
| Token streaming | `onToken` callback | Immutably updates assistant message: `messages.map { if (it.id == requestId) it.copy(text = it.text + tokenPiece) else it }`. |
| Completion | `onDone` callback | Sets final text, TPS, `generating = false`. |
| Auto-scroll | `listState.animateScrollToItem` | Called after each token and on send. |

### 4.5 Stop Generation

| Step | Where | Detail |
|---|---|---|
| Trigger | Same circular button, now showing a close icon | Visible when `generating == true`. |
| Action | `OneBitNativeBridge.stop()` | Sets `g_cancel.store(true)`. |
| Result | Native loop exits; calls `onDone` with whatever was generated so far. |

### 4.6 Error Handling in Chat

| Step | Where | Detail |
|---|---|---|
| Native errors | Template, tokenize, or decode failures in native | `cb_error` dispatched. |
| Kotlin handler | `onError` callback | Sets `text = "Error: $message"` and `isError = true` on the assistant message. |
| Visual | Chat bubble | Error messages are rendered in red (`Color(0xFFFF5252)`). |

### 4.7 TPS Display

| Step | Where | Detail |
|---|---|---|
| Calculation | Native `onDone` | `tps = generatedTokens / elapsedSeconds`. |
| Display | Top app bar, next to the "Advanced" badge | Shows `"XX.XX TPS"` in grey. Hides when no generation has run (`tps == null`). |

### 4.8 Cleanup

| Step | Where | Detail |
|---|---|---|
| Trigger | `DisposableEffect(Unit).onDispose` | Fires when the composable leaves composition (activity destroyed). |
| Action | `stop()` + `unloadModel()` | Cancels any running generation and frees native resources. |

---

## 5. Function Coupling Map

```
MainActivity.onCreate(bundle)
│
├─ setContent
│   └─ MaterialTheme(darkColorScheme(#131314))
│       └─ Surface(fillMaxSize)
│           └─ OneBitRoot(onPickModel)
│               │
│               ├─ [State] modelPath, status, importing, input,
│               │           messages, generating, isModelLoaded, tps
│               │
│               ├─ LaunchedEffect(Unit)          ──┐
│               │   └─ existingModelPath(ctx)       │ auto-load
│               │       └─ initBackend() ───────────┤ flow
│               │       └─ loadModel(path,2048,8) ──┘
│               │
│               ├─ DisposableEffect(Unit)         ──┐
│               │   └─ onDispose                    │ cleanup
│               │       ├─ stop()                   │ flow
│               │       └─ unloadModel()          ──┘
│               │
│               ├─ TopAppBar
│               │   ├─ Clear button toggle
│               │   │   └─ scope.launch → initBackend() / unloadModel()
│               │   └─ Menu → picker.launch()
│               │       └─ onPickModel(uri)
│               │           └─ importModelAndLoad()
│               │               ├─ Dispatchers.IO: copy file
│               │               └─ Dispatchers.Default: initBackend() + loadModel()
│               │
│               ├─ LazyColumn (chat messages)
│               │   └─ items(messages, key = it.id)
│               │       ├─ role="user"   → right-aligned bubble, dark bg
│               │       └─ role="assistant" → left-aligned, logo avatar,
│               │           spinner when empty+generating, red if isError
│               │
│               └─ Input Row
│                   ├─ BasicTextField (input, onValueChange)
│                   └─ Send/Stop button
│                       ├─ [Stop] → OneBitNativeBridge.stop()
│                       └─ [Send] →
│                           ├─ messages += [ChatMsg(user), ChatMsg(assistant, id=req)]
│                           ├─ OneBitCallback
│                           │   ├─ onToken  → messages.map { copy(text += piece) }
│                           │   ├─ onDone   → messages.map { copy(text) }, tps, generating=false
│                           │   └─ onError  → messages.map { copy(text, isError=true) }, generating=false
│                           └─ OneBitNativeBridge.generate(req, prompt, 512, 0.8, 0.95, 40, cb)
│
└─ lifecycleScope (coroutine management for model import)
```

### 5.1 Native Generation Flow (expanded)

```
OneBitNativeBridge_generate(JNIEnv*, jclass, jRequestId, jPrompt, maxTokens,
                            temperature, topP, topK, jobject callback)
│
├─ Verify callback != null, model/context loaded (under g_mu)
├─ Cache JNI method IDs (g_cbClass, g_onToken, g_onDone, g_onError) — once
├─ Create global ref to callback object (cbGlobal)
├─ g_cancel.store(false)
│
└─ std::thread([...] detach):
    ├─ AttachCurrentThread (get JNIEnv for this native thread)
    ├─ Copy model/context pointers under g_mu
    │
    ├─ Create sampler chain: top_k → top_p → temp → dist
    │
    ├─ llama_chat_apply_template([{"user", prompt}], add_ass=true)
    ├─ llama_tokenize → tokens[]
    │
    ├─ Full-prompt batch decode (llama_decode with all tokens)
    │   └─ Fallback: single-token decode if batch fails
    │
    ├─ [Loop] while !g_cancel && generated < maxTokens:
    │   ├─ llama_sampler_sample(smpl, ctx, -1)
    │   ├─ llama_sampler_accept(smpl, id)
    │   ├─ BREAK if id == EOS or EOT
    │   ├─ llama_token_to_piece (special=true for detection, false for display)
    │   ├─ Stop-marker scan against (out + pieceSearch)
    │   │   Markers: "\nUser:", "\nAssistant:", "User:", "Assistant:",
    │   │            "\nResponse:", "<|user|>", "<|assistant|>",
    │   │            "<|endofturn|>", "<|eot|>", "<|end|>", "<|endoftext|>"
    │   ├─ If stop marker found: trim, emit delta, BREAK
    │   ├─ cb_token(cbGlobal, reqId, pieceOut)
    │   ├─ llama_decode (single-token batch for next position)
    │   └─ generated++
    │
    ├─ Calculate TPS = generated / (elapsed_us / 1e6)
    ├─ Final safety trim (same stop markers)
    ├─ cb_done(cbGlobal, reqId, out, generated, tps)
    ├─ llama_sampler_free(smpl)
    └─ DetachCurrentThread
```

---

## 6. State Machine

The model/context has four meaningful states tracked by `isModelLoaded` + `generating`:

```
                    ┌─────────────┐
          app start │  UNLOADED   │
          ┌───────►│             │◄──────── load fails
          │        └──────┬──────┘
          │               │ loadModel()
          │               ▼
          │        ┌─────────────┐
          │        │   LOADING   │
          │        │             │
          │        └──────┬──────┘
          │               │ success
          │               ▼
          │        ┌─────────────┐    unloadModel()    ┌─────────────┐
          │        │   LOADED    │────────────────────►│  UNLOADED   │
          │        │             │◄────────────────────│             │
          │        └──────┬──────┘                     └─────────────┘
          │               │ generate()
          │               ▼
          │        ┌─────────────┐    onDone()/onError()
          │        │ GENERATING  │────────────────────►┌─────────────┐
          │        │             │◄── stop() (cancel)   │   LOADED    │
          │        └─────────────┘                      └─────────────┘
```

Transitions:
- **UNLOADED → LOADING**: `loadModel()` called from auto-load, manual toggle, or import.
- **LOADING → LOADED**: `llama_load_model_from_file` + `llama_new_context_with_model` succeed.
- **LOADING → UNLOADED**: Exception thrown (bad file path, incompatible model).
- **LOADED → GENERATING**: User sends a message → `generate()` spawns native thread.
- **GENERATING → LOADED**: `onDone` or `onError` callback fires (normal completion).
- **GENERATING → LOADED** (via cancel): `stop()` sets `g_cancel=true` → loop exits → `onDone` fires with partial output.
- **LOADED → UNLOADED**: User presses unload toggle, or `onDispose` in cleanup.

---

## 7. Threading Model

### 7.1 Kotlin Threads

| Operation | Dispatcher | Reason |
|---|---|---|
| File copy (model import) | `Dispatchers.IO` | File I/O, blocking. |
| Model load/unload | `Dispatchers.Default` | CPU-bound, may be slow. |
| UI state updates (callbacks) | Compose scope (main thread) | Required by Compose for state mutation. |

### 7.2 Native Threads

| Thread | Purpose | Lifetime |
|---|---|---|
| Main thread (JNI calls) | `initBackend`, `loadModel`, `unloadModel`, `stop`, `ping` | Synchronous, returns immediately. |
| Generation thread | `std::thread` spawned by `generate()`, detached | Lives until generation completes or is cancelled. Attaches/detaches JNI env. |
| `g_mu` mutex | Serializes model/context/sampler access | Held during `loadModel`, `unloadModel`, and `generate` preamble. |

The generation thread does **not** hold `g_mu` during the loop — the mutex is released after copying pointers. This allows `stop()` (atomic store) and `unloadModel()` (mutex acquire) to run without deadlocking.

---

## 8. Native Build & Dependencies

### 8.1 CMake Structure

```
app/src/main/cpp/CMakeLists.txt
├─ add_library(onebit SHARED onebit_jni.cpp)
├─ target_compile_options(onebit PRIVATE -O3 -DNDEBUG)
├─ target_include_directories(onebit PRIVATE
│      ${ONEBIT_BITNET_DIR}/3rdparty/llama.cpp/include
│      ${ONEBIT_BITNET_DIR}/3rdparty/llama.cpp/ggml/include)
├─ set(BITNET_ARM_TL1 ON)          # BitNet ARM tile-level-1 optimization
├─ add_subdirectory(${ONEBIT_BITNET_DIR})   # Builds llama + ggml from external fork
└─ target_link_libraries(onebit llama ggml android log)
```

The external BitNet fork path is hardcoded: `/Users/anirudhmalik/Desktop/onebit/onebit-android/bitnet`. It must contain llama.cpp source with the BitNet ARM patches.

### 8.2 Kotlin Dependencies (from `app/build.gradle.kts`)

| Dependency | Version | Purpose |
|---|---|---|
| `androidx.core:core-ktx` | 1.15.0 | Kotlin extensions for Android APIs |
| `androidx.activity:activity-compose` | 1.9.3 | Compose Activity integration |
| `compose-bom` | 2024.12.01 | Compose version alignment |
| `compose.ui`, `material3`, `ui-tooling-preview` | BOM | UI toolkit |
| `lifecycle-runtime-ktx` | 2.8.7 | Lifecycle-aware coroutines |
| `lifecycle-viewmodel-compose` | 2.8.7 | Declared but **unused** in current code |
| `kotlinx-coroutines-android` | 1.8.1 | Coroutines on Android |

### 8.3 Build Configuration

| Setting | Value |
|---|---|
| compileSdk / targetSdk | 36 |
| minSdk | 24 |
| ABI filter | `arm64-v8a` only |
| Kotlin JVM target | Java 17 |
| C++ standard | C++17 |
| Gradle | 8.13 |
| AGP | 8.7.3 |
| Kotlin | 2.1.20 |

---

## 9. Key Design Decisions & Rationale

### 9.1 Single-Activity, Single-Composable
No navigation, no routes. The entire UI is `OneBitRoot`. Rationale: the app has one screen (chat) with no secondary views. Simplicity over premature abstraction.

### 9.2 No ViewModel
State lives in `remember { mutableStateOf }` locals inside the composable. The ViewModel Compose dependency is declared but unused. Rationale: a single screen with no configuration-change survival requirements doesn't benefit from ViewModel overhead.

### 9.3 Immutable Message Updates
Messages are **not** `MutableStateList`. They use `messages = messages.map { ... }` with `copy()`. Rationale: Compose recomposition relies on state object identity changes; immutable list replacement triggers recomposition correctly, and `copy()` is idiomatic for data classes.

### 9.4 Full-Prompt Batch Decode
The native engine decodes all prompt tokens in a single `llama_decode` call rather than one-by-one. Rationale: the BitNet architecture supports parallel prompt processing. A fallback to single-token decode exists if the batch fails.

### 9.5 Stop-Marker Detection in Native Code
Instead of relying solely on EOS/EOT tokens, the native loop scans accumulated output for chat-template markers like `User:`, `Assistant:`, etc. Rationale: prevents the model from "roleplaying" the user or continuing the conversation past its turn.

### 9.6 `singleTask` Launch Mode
The activity uses `launchMode="singleTask"`. Rationale: prevents multiple instances of the activity from stacking, which would create duplicate native model loads.

### 9.7 Hardcoded Model Filename
The imported model is always saved as `ggml-model-i2_s.gguf`. There is no support for multiple model files. Rationale: prototype stage; the BitNet model format uses this naming convention.

---

## 10. Limitations & Gaps

| Area | Gap |
|---|---|
| **Testing** | No unit tests, no UI tests, no native tests. |
| **Error Recovery** | Load failures show a status string but offer no retry UX besides re-toggling. |
| **Hardcoded Paths** | External BitNet fork path is an absolute path on one developer's machine. |
| **Hardcoded Params** | Thread count (8), context size (2048), and sampling params are constants in Kotlin code. |
| **Single Model** | Only one model file supported, always named `ggml-model-i2_s.gguf`. |
| **No DI** | Tight coupling between Activity, Bridge, and Compose state — hard to unit test. |
| **No Configuration Persistence** | Model path, message history, and settings are lost on process death. |
| **No Logging Abstraction** | Native uses `__android_log_print` directly; Kotlin uses no logger. |
| **ABI Lock** | Only `arm64-v8a` — no x86_64 emulator support for debugging. |
