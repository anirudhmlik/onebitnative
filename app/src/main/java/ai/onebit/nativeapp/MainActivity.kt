package ai.onebit.nativeapp

import android.content.Context
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.Image
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Star
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.util.UUID

class MainActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContent {
      MaterialTheme(
        colorScheme = darkColorScheme(
          background = Color(0xFF131314),
          surface = Color(0xFF131314)
        )
      ) {
        Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
          OneBitRoot(
            onPickModel = { uri ->
              lifecycleScope.launch {
                importModelAndLoad(uri)
              }
            },
          )
        }
      }
    }
  }

  private suspend fun importModelAndLoad(uri: Uri) {
    withContext(Dispatchers.IO) {
      val destDir = File(filesDir, "models").apply { mkdirs() }
      val dest = File(destDir, "ggml-model-i2_s.gguf")
      contentResolver.openInputStream(uri).use { input ->
        requireNotNull(input) { "Failed to open model input stream" }
        FileOutputStream(dest).use { out ->
          input.copyTo(out)
        }
      }
      withContext(Dispatchers.Default) {
        OneBitNativeBridge.initBackend()
        OneBitNativeBridge.loadModel(dest.absolutePath, /*nCtx=*/2048, /*nThreads=*/8)
      }
    }
  }
}

private data class ChatMsg(val id: String, val role: String, val text: String, val isError: Boolean = false)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun OneBitRoot(
  onPickModel: (Uri) -> Unit,
) {
  val ctx = LocalContext.current
  var modelPath by remember { mutableStateOf<String?>(existingModelPath(ctx)) }
  var status by remember { mutableStateOf("Ready") }
  var importing by remember { mutableStateOf(false) }
  var input by remember { mutableStateOf("") }
  var messages by remember { mutableStateOf(listOf<ChatMsg>(
    ChatMsg("welcome", "assistant", "Hello! I am OneBit. How can I help you today?")
  )) }
  var generating by remember { mutableStateOf(false) }
  var tps by remember { mutableStateOf<Float?>(null) }
  val scope = rememberCoroutineScope()
  val listState = rememberLazyListState()

  val picker =
    rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
      if (uri != null) {
        importing = true
        status = "Importing model…"
        onPickModel(uri)
      }
    }

  LaunchedEffect(Unit) {
    val path = existingModelPath(ctx)
    modelPath = path
    if (path != null) {
      status = "Loading BitNet Model into Memory..."
      withContext(Dispatchers.Default) {
        try {
          OneBitNativeBridge.initBackend()
          OneBitNativeBridge.loadModel(path, 2048, 8)
          status = "BitNet Model Loaded"
        } catch (e: Exception) {
          status = "Failed to load model: ${e.message}"
        }
      }
      importing = false
    } else {
      status = "Missing Model"
    }
  }

  DisposableEffect(Unit) {
    onDispose {
      try {
        OneBitNativeBridge.stop()
        OneBitNativeBridge.unloadModel()
      } catch (_: Throwable) {}
    }
  }

  Scaffold(
    topBar = {
      TopAppBar(
        title = {
          Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
              "OneBit",
              fontSize = 20.sp,
              fontWeight = FontWeight.Medium,
              color = Color(0xFFE3E3E3)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Surface(
              shape = RoundedCornerShape(4.dp),
              color = Color(0xFF282A2C),
              modifier = Modifier.padding(top = 2.dp)
            ) {
              Text(
                "Advanced",
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                color = Color(0xFFD4A5FF),
                modifier = Modifier.padding(horizontal = 4.dp, vertical = 2.dp)
              )
            }
            if (tps != null) {
              Spacer(modifier = Modifier.width(8.dp))
              Surface(
                shape = RoundedCornerShape(4.dp),
                color = Color.Transparent,
                modifier = Modifier.padding(top = 2.dp)
              ) {
                Text(
                  "${"%.2f".format(tps)} TPS",
                  fontSize = 10.sp,
                  fontWeight = FontWeight.Bold,
                  color = Color(0xFF9AA0A6)
                )
              }
            }
          }
        },
        actions = {
          var menuExpanded by remember { mutableStateOf(false) }
          IconButton(onClick = { menuExpanded = true }) {
            Icon(Icons.Default.MoreVert, contentDescription = "Menu", tint = Color(0xFFE3E3E3))
          }
          DropdownMenu(
            expanded = menuExpanded,
            onDismissRequest = { menuExpanded = false },
            modifier = Modifier.background(Color(0xFF282A2C))
          ) {
            DropdownMenuItem(
              text = { Text(if (importing) "Importing..." else "Import .gguf Model", color = Color.White) },
              onClick = {
                menuExpanded = false
                if (!importing) picker.launch(arrayOf("*/*"))
              }
            )
          }
        },
        colors = TopAppBarDefaults.topAppBarColors(
          containerColor = Color(0xFF131314)
        )
      )
    },
    containerColor = Color(0xFF131314)
  ) { paddingValues ->
    Column(
      modifier = Modifier
        .fillMaxSize()
        .padding(paddingValues)
    ) {
      
      // Chat Messages List
      LazyColumn(
        state = listState,
        modifier = Modifier
          .weight(1f)
          .fillMaxWidth(),
        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(24.dp)
      ) {
        items(messages, key = { it.id }) { m ->
          val isUser = m.role == "user"
          
          if (isUser) {
            Row(
              modifier = Modifier.fillMaxWidth(),
              horizontalArrangement = Arrangement.End
            ) {
              Box(
                modifier = Modifier
                  .widthIn(max = 280.dp)
                  .clip(RoundedCornerShape(20.dp, 20.dp, 4.dp, 20.dp))
                  .background(Color(0xFF282A2C))
                  .padding(horizontal = 16.dp, vertical = 12.dp)
              ) {
                Text(
                  text = m.text,
                  color = Color(0xFFE3E3E3),
                  fontSize = 16.sp,
                  lineHeight = 24.sp
                )
              }
            }
          } else {
            Row(
              modifier = Modifier.fillMaxWidth(),
              verticalAlignment = Alignment.Top,
              horizontalArrangement = Arrangement.Start
            ) {
              Box(
                modifier = Modifier
                  .size(32.dp)
                  .clip(CircleShape)
                  .background(Color.White),
                contentAlignment = Alignment.Center
              ) {
                Image(
                  painter = painterResource(id = R.drawable.logo),
                  contentDescription = "Assistant Logo",
                  modifier = Modifier.size(32.dp).clip(CircleShape)
                )
              }
              Spacer(modifier = Modifier.width(12.dp))
              Column(modifier = Modifier.weight(1f)) {
                if (m.text.isEmpty() && generating && messages.last() == m) {
                  Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.height(24.dp)) {
                    CircularProgressIndicator(
                      color = Color(0xFFD4A5FF),
                      strokeWidth = 2.dp,
                      modifier = Modifier.size(16.dp)
                    )
                  }
                } else {
                  Text(
                    text = m.text,
                    color = if (m.isError) Color(0xFFFF5252) else Color(0xFFE3E3E3),
                    fontSize = 16.sp,
                    lineHeight = 26.sp
                  )
                }
              }
            }
          }
        }
      }

      // Model missing warning
      if (modelPath == null) {
        Box(modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp), contentAlignment = Alignment.Center) {
          Text(
            text = "Please import a BitNet model via the top-right menu to chat.",
            color = Color(0xFF9AA0A6),
            fontSize = 12.sp
          )
        }
      }

      // Input Field Area
      Row(
        modifier = Modifier
          .fillMaxWidth()
          .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment = Alignment.Bottom,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
      ) {
        Box(
          modifier = Modifier
            .weight(1f)
            .clip(RoundedCornerShape(24.dp))
            .background(Color(0xFF1E1F22))
            .padding(horizontal = 16.dp, vertical = 12.dp)
        ) {
          if (input.isEmpty()) {
            Text(
              text = "Ask OneBit...",
              color = Color(0xFF9AA0A6),
              fontSize = 16.sp,
              modifier = Modifier.padding(vertical = 2.dp)
            )
          }
          BasicTextField(
            value = input,
            onValueChange = { input = it },
            enabled = !generating && modelPath != null,
            textStyle = TextStyle(
              color = Color(0xFFE3E3E3),
              fontSize = 16.sp,
              lineHeight = 24.sp
            ),
            cursorBrush = SolidColor(Color(0xFFD4A5FF)),
            modifier = Modifier
              .fillMaxWidth()
              .defaultMinSize(minHeight = 24.dp)
          )
        }

        Box(
          modifier = Modifier
            .size(48.dp)
            .clip(CircleShape)
            .background(if (generating) Color(0xFFE53935) else if (input.isNotBlank() && modelPath != null) Color(0xFFE3E3E3) else Color(0xFF282A2C)),
          contentAlignment = Alignment.Center
        ) {
          IconButton(
            onClick = {
              if (generating) {
                OneBitNativeBridge.stop()
                generating = false
                return@IconButton
              }
              val prompt = input.trim()
              if (prompt.isEmpty() || modelPath == null) return@IconButton
              input = ""
              val req = UUID.randomUUID().toString()
              generating = true
              tps = null
              messages = messages + ChatMsg(UUID.randomUUID().toString(), "user", prompt) +
                ChatMsg(req, "assistant", "")

              scope.launch {
                listState.animateScrollToItem(messages.size - 1)
              }

              val cb =
                object : OneBitCallback {
                  override fun onToken(requestId: String, tokenPiece: String) {
                    scope.launch {
                      messages =
                        messages.map { if (it.id == requestId) it.copy(text = it.text + tokenPiece) else it }
                      listState.animateScrollToItem(messages.size - 1)
                    }
                  }

                  override fun onDone(requestId: String, fullText: String, tokens: Int, tpsValue: Float) {
                    scope.launch {
                      generating = false
                      tps = tpsValue
                      messages =
                        messages.map { if (it.id == requestId) it.copy(text = fullText) else it }
                    }
                  }

                  override fun onError(requestId: String, message: String) {
                    scope.launch {
                      generating = false
                      messages =
                        messages.map { if (it.id == requestId) it.copy(text = "Error: $message", isError = true) else it }
                    }
                  }
                }

              OneBitNativeBridge.generate(
                requestId = req,
                prompt = prompt,
                maxTokens = 512,
                temperature = 0.8f,
                topP = 0.95f,
                topK = 40,
                callback = cb,
              )
            },
            enabled = generating || (input.isNotBlank() && modelPath != null)
          ) {
            if (generating) {
              Icon(
                Icons.Default.Close,
                contentDescription = "Stop",
                tint = Color.White,
                modifier = Modifier.size(24.dp)
              )
            } else {
              Icon(
                Icons.Default.Send,
                contentDescription = "Send",
                tint = if (input.isNotBlank() && modelPath != null) Color(0xFF131314) else Color(0xFF5F6368),
                modifier = Modifier.size(20.dp).offset(x = 2.dp)
              )
            }
          }
        }
      }
      
      Spacer(modifier = Modifier.windowInsetsBottomHeight(WindowInsets.navigationBars))
    }
  }
}

private fun existingModelPath(ctx: Context): String? {
  val f = File(ctx.filesDir, "models/ggml-model-i2_s.gguf")
  return if (f.exists() && f.isFile && f.length() > 1_000_000_000L) f.absolutePath else null
}
