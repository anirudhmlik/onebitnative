#include <jni.h>
#include <android/log.h>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "llama.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "OneBitNative", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "OneBitNative", __VA_ARGS__)

static JavaVM * g_vm = nullptr;

static std::mutex g_mu;
static llama_model * g_model = nullptr;
static llama_context * g_ctx = nullptr;
static llama_sampler * g_smpl = nullptr;
static std::atomic<bool> g_cancel{false};

static jclass g_cbClass = nullptr;
static jmethodID g_onToken = nullptr;
static jmethodID g_onDone  = nullptr;
static jmethodID g_onError = nullptr;

static void throw_java(JNIEnv * env, const char * msg) {
    jclass ex = env->FindClass("java/lang/RuntimeException");
    if (ex) env->ThrowNew(ex, msg);
}

static std::string j2s(JNIEnv * env, jstring js) {
    if (!js) return {};
    const char * c = env->GetStringUTFChars(js, nullptr);
    std::string s = c ? c : "";
    if (c) env->ReleaseStringUTFChars(js, c);
    return s;
}

extern "C" JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM * vm, void *) {
    g_vm = vm;
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_onebit_nativeapp_OneBitNativeBridge_ping(JNIEnv * env, jclass) {
    LOGI("ping()");
    return env->NewStringUTF("pong");
}

extern "C" JNIEXPORT void JNICALL
Java_ai_onebit_nativeapp_OneBitNativeBridge_initBackend(JNIEnv *, jclass) {
    std::lock_guard<std::mutex> lock(g_mu);
    llama_backend_init();
}

extern "C" JNIEXPORT void JNICALL
Java_ai_onebit_nativeapp_OneBitNativeBridge_loadModel(JNIEnv * env, jclass, jstring jPath, jint nCtx, jint nThreads) {
    const std::string path = j2s(env, jPath);
    if (path.empty()) {
        throw_java(env, "modelPath is empty");
        return;
    }

    std::lock_guard<std::mutex> lock(g_mu);

    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_free_model(g_model);
        g_model = nullptr;
    }
    if (g_smpl) {
        llama_sampler_free(g_smpl);
        g_smpl = nullptr;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = true;
    mparams.use_mlock = false;

    g_model = llama_load_model_from_file(path.c_str(), mparams);
    if (!g_model) {
        throw_java(env, "llama_load_model_from_file returned null");
        return;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) nCtx;
    cparams.n_threads = (int32_t) nThreads;
    cparams.n_threads_batch = (int32_t) nThreads;

    g_ctx = llama_new_context_with_model(g_model, cparams);
    if (!g_ctx) {
        llama_free_model(g_model);
        g_model = nullptr;
        throw_java(env, "llama_new_context_with_model returned null");
        return;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_ai_onebit_nativeapp_OneBitNativeBridge_unloadModel(JNIEnv *, jclass) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_smpl) { llama_sampler_free(g_smpl); g_smpl = nullptr; }
    if (g_ctx)   { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_free_model(g_model); g_model = nullptr; }
}

extern "C" JNIEXPORT void JNICALL
Java_ai_onebit_nativeapp_OneBitNativeBridge_stop(JNIEnv *, jclass) {
    g_cancel.store(true);
}

static void cb_error(JNIEnv * env, jobject cb, const std::string & requestId, const std::string & msg) {
    jstring jReq = env->NewStringUTF(requestId.c_str());
    jstring jMsg = env->NewStringUTF(msg.c_str());
    env->CallVoidMethod(cb, g_onError, jReq, jMsg);
    env->DeleteLocalRef(jReq);
    env->DeleteLocalRef(jMsg);
}

static void cb_token(JNIEnv * env, jobject cb, const std::string & requestId, const std::string & piece) {
    jstring jReq = env->NewStringUTF(requestId.c_str());
    jstring jTok = env->NewStringUTF(piece.c_str());
    env->CallVoidMethod(cb, g_onToken, jReq, jTok);
    env->DeleteLocalRef(jReq);
    env->DeleteLocalRef(jTok);
}

static void cb_done(JNIEnv * env, jobject cb, const std::string & requestId, const std::string & full, int tokens, float tps) {
    jstring jReq = env->NewStringUTF(requestId.c_str());
    jstring jTxt = env->NewStringUTF(full.c_str());
    env->CallVoidMethod(cb, g_onDone, jReq, jTxt, (jint) tokens, (jfloat) tps);
    env->DeleteLocalRef(jReq);
    env->DeleteLocalRef(jTxt);
}

extern "C" JNIEXPORT void JNICALL
Java_ai_onebit_nativeapp_OneBitNativeBridge_generate(
        JNIEnv * env,
        jclass,
        jstring jRequestId,
        jstring jPrompt,
        jint maxTokens,
        jfloat temperature,
        jfloat topP,
        jint topK,
        jobject callback) {
    const std::string requestId = j2s(env, jRequestId);
    const std::string prompt    = j2s(env, jPrompt);

    if (!callback) {
        throw_java(env, "callback is null");
        return;
    }

    {
        std::lock_guard<std::mutex> lock(g_mu);
        if (!g_ctx || !g_model) {
            throw_java(env, "model not loaded");
            return;
        }

        if (!g_cbClass) {
            jclass local = env->GetObjectClass(callback);
            g_cbClass = (jclass) env->NewGlobalRef(local);
            g_onToken = env->GetMethodID(g_cbClass, "onToken", "(Ljava/lang/String;Ljava/lang/String;)V");
            g_onDone  = env->GetMethodID(g_cbClass, "onDone",  "(Ljava/lang/String;Ljava/lang/String;IF)V");
            g_onError = env->GetMethodID(g_cbClass, "onError", "(Ljava/lang/String;Ljava/lang/String;)V");
            env->DeleteLocalRef(local);
            if (!g_onToken || !g_onDone || !g_onError) {
                throw_java(env, "Failed to look up callback methods");
                return;
            }
        }
    }

    jobject cbGlobal = env->NewGlobalRef(callback);
    g_cancel.store(false);

    std::thread([requestId, prompt, maxTokens, temperature, topP, topK, cbGlobal]() {
        JNIEnv * tenv = nullptr;
        bool attached = false;
        if (g_vm->GetEnv((void **) &tenv, JNI_VERSION_1_6) != JNI_OK) {
            if (g_vm->AttachCurrentThread(&tenv, nullptr) != JNI_OK) {
                return;
            }
            attached = true;
        }

        auto cleanup = [&]() {
            if (tenv && cbGlobal) {
                tenv->DeleteGlobalRef(cbGlobal);
            }
            if (attached) {
                g_vm->DetachCurrentThread();
            }
        };

        llama_context * ctx = nullptr;
        llama_model * model = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_mu);
            ctx = g_ctx;
            model = g_model;
        }
        if (!ctx || !model) {
            cb_error(tenv, cbGlobal, requestId, "model not loaded");
            cleanup();
            return;
        }

        // Create a fresh sampler chain per request
        llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
        llama_sampler * smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(topK));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(topP, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        const int64_t t0 = llama_time_us();

        std::vector<llama_token> tokens(8192);
        std::string chatml_prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        const int n_prompt = llama_tokenize(model, chatml_prompt.c_str(), (int) chatml_prompt.size(), tokens.data(), (int) tokens.size(), true, true);
        if (n_prompt < 0) {
            cb_error(tenv, cbGlobal, requestId, "llama_tokenize failed");
            llama_sampler_free(smpl);
            cleanup();
            return;
        }
        tokens.resize((size_t) n_prompt);

        // Evaluate prompt token-by-token for BitNet ARM TL1 limitation
        llama_batch batch = llama_batch_init(1, 0, 1);
        for (int i = 0; i < (int) tokens.size(); i++) {
            batch.token[0] = tokens[i];
            batch.pos[0] = i;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = (i == (int) tokens.size() - 1);
            batch.n_tokens = 1;
            
            if (llama_decode(ctx, batch) != 0) {
                cb_error(tenv, cbGlobal, requestId, "llama_decode(prompt) failed");
                llama_batch_free(batch);
                llama_sampler_free(smpl);
                cleanup();
                return;
            }
        }

        llama_batch_free(batch);

        std::string out;
        out.reserve(4096);
        int generated = 0;
        int pos = (int) tokens.size();

        while (!g_cancel.load() && generated < maxTokens) {
            const llama_token id = llama_sampler_sample(smpl, ctx, -1);
            llama_sampler_accept(smpl, id);

            if (id == llama_token_eos(model)) {
                break;
            }

            char piece[256];
            const int n = llama_token_to_piece(model, id, piece, sizeof(piece), 0, true);
            if (n > 0) {
                out.append(piece, piece + n);
                cb_token(tenv, cbGlobal, requestId, std::string(piece, piece + n));
            }

            // Decode next token
            llama_batch b = llama_batch_init(1, 0, 1);
            b.token[0] = id;
            b.pos[0] = pos++;
            b.n_seq_id[0] = 1;
            b.seq_id[0][0] = 0;
            b.logits[0] = true;
            b.n_tokens = 1;
            if (llama_decode(ctx, b) != 0) {
                llama_batch_free(b);
                cb_error(tenv, cbGlobal, requestId, "llama_decode(next) failed");
                llama_sampler_free(smpl);
                cleanup();
                return;
            }
            llama_batch_free(b);

            generated++;
        }

        const int64_t t1 = llama_time_us();
        const double sec = (t1 - t0) / 1e6;
        const float tps = sec > 0 ? (float) (generated / sec) : 0.0f;
        cb_done(tenv, cbGlobal, requestId, out, generated, tps);

        llama_sampler_free(smpl);
        cleanup();
    }).detach();
}

