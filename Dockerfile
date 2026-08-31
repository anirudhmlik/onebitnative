FROM eclipse-temurin:17-jdk-jammy AS build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget unzip git ninja-build \
    && rm -rf /var/lib/apt/lists/*

ENV ANDROID_SDK_ROOT=/opt/android-sdk
ENV PATH="${ANDROID_SDK_ROOT}/cmdline-tools/latest/bin:${PATH}"

RUN mkdir -p ${ANDROID_SDK_ROOT} && \
    wget -q https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip -O /tmp/cmdline-tools.zip && \
    unzip -q /tmp/cmdline-tools.zip -d ${ANDROID_SDK_ROOT} && \
    mv ${ANDROID_SDK_ROOT}/cmdline-tools ${ANDROID_SDK_ROOT}/_latest && \
    mkdir -p ${ANDROID_SDK_ROOT}/cmdline-tools && \
    mv ${ANDROID_SDK_ROOT}/_latest ${ANDROID_SDK_ROOT}/cmdline-tools/latest && \
    rm /tmp/cmdline-tools.zip

RUN yes | sdkmanager --sdk_root=${ANDROID_SDK_ROOT} \
    "platforms;android-36" \
    "build-tools;36.0.0" \
    "ndk;27.3.13750724" \
    "cmake;3.22.1"

ARG BITNET_REPO_URL
ARG BITNET_REPO_BRANCH=main

RUN if [ -n "${BITNET_REPO_URL}" ]; then \
      git clone --depth 1 --branch "${BITNET_REPO_BRANCH}" "${BITNET_REPO_URL}" /opt/bitnet; \
    fi

COPY . /app
WORKDIR /app

RUN if [ -d /opt/bitnet ]; then \
      sed -i 's|/Users/anirudhmalik/Desktop/onebit/onebit-android/bitnet|/opt/bitnet|g' app/src/main/cpp/CMakeLists.txt; \
    fi

RUN echo "sdk.dir=${ANDROID_SDK_ROOT}" > local.properties

RUN --network=host ./gradlew assembleDebug --no-daemon

FROM scratch AS final
COPY --from=build /app/app/build/outputs/apk/debug/app-debug.apk /app-debug.apk
