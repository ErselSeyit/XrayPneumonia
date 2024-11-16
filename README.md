# X_Ray_P

This project is an Android application developed using Flutter. 

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have installed [Android Studio](https://developer.android.com/studio/install). Follow the instructions on the website to download and install it depending on your operating system.
- You have installed [Flutter](https://docs.flutter.dev/get-started/install). Follow the instructions on the website to download and install it depending on your operating system.
- You have added necessary extensions to your IDE of choice.

## Checking the Installation

To verify that everything is set up correctly, run `flutter doctor -v` and locate the path printed after 'Java binary at:'. Then use that fully qualified path replacing `java` (at the end) with `keytool`. If your path includes space-separated names, such as Program Files, use platform-appropriate notation for the names. For example, on Mac/Linux use `Program\ Files`, and on Windows use `"Program Files"`.

## Getting Package Dependencies

To fetch the necessary package dependencies for the app, run `flutter pub get` from the terminal.

## Running the App

This section describes how to build a release app bundle. If you completed the signing steps, the app bundle will be signed. At this point, you might consider obfuscating your Dart code to make it more difficult to reverse engineer. Obfuscating your code involves adding a couple flags to your build command, and maintaining additional files to de-obfuscate stack traces.

From the command line:

1. Navigate to your project directory with `cd [project]`.
2. Run `flutter build appbundle`. (Running `flutter build` defaults to a release build.) The release bundle for your app is created at `[project]/build/app/outputs/bundle/release/app.aab`.

By default, the app bundle contains your Dart code and the Flutter runtime compiled for armeabi-v7a (ARM 32-bit), arm64-v8a (ARM 64-bit), and x86-64 (x86 64-bit).

## Building an APK

To build an APK:

From the command line:

1. Navigate to your project directory with `cd [project]`.
2. Run `flutter build apk --split-per-abi`. (The `flutter build` command defaults to `--release`.)

## Installing an APK on a Device

To install the APK on a connected Android device:

From the command line:

1. Connect your Android device to your computer with a USB cable.
2. Navigate to your project directory with `cd [project]`.
3. Run `flutter install`.