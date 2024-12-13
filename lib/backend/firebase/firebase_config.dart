import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/foundation.dart';

Future initFirebase() async {
  if (kIsWeb) {
    await Firebase.initializeApp(
        options: const FirebaseOptions(
            apiKey: "AIzaSyB1OMOTaes2xLuWP8ZjMFgm4hR4fjM036Y",
            authDomain: "crear-producto-cod7xu.firebaseapp.com",
            projectId: "crear-producto-cod7xu",
            storageBucket: "crear-producto-cod7xu.firebasestorage.app",
            messagingSenderId: "13526775481",
            appId: "1:13526775481:web:06f25f9e1069ae7bcdcd68"));
  } else {
    await Firebase.initializeApp();
  }
}
