# Backend WebSocket pour la reconnaissance de langue des signes
# ============================================================

import json
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Sign Language Recognition API",
    description="WebSocket API pour recevoir des frames vidéo et renvoyer les mots reconnus",
    version="1.0.0"
)

# Autoriser les connexions depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, mettre l'URL exacte du frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def recognize_sign(frame_data: str) -> str | None:
    """
    🧠 FONCTION À IMPLÉMENTER
    
    Cette fonction reçoit une image en base64 et doit retourner le mot reconnu.
    
    Args:
        frame_data: Image en base64 (format: "data:image/jpeg;base64,/9j/4AAQ...")
    
    Returns:
        Le mot reconnu (str) ou None si rien n'est détecté
    
    Exemple pour extraire l'image:
        # Enlever le préfixe "data:image/jpeg;base64,"
        base64_string = frame_data.split(",")[1]
        # Décoder en bytes
        image_bytes = base64.b64decode(base64_string)
        # Convertir en numpy array avec OpenCV
        import numpy as np
        import cv2
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    """
    # ============================================
    # TODO: Remplacer par ton modèle ML
    # ============================================
    
    # Mock pour tester la connexion (renvoie "Hello" toutes les 2 secondes environ)
    import random
    if random.random() < 0.03:  # ~1 fois toutes les 30 frames (2 sec à 15 FPS)
        words = ["Hello", "Thank you", "Please", "Yes", "No", "Help"]
        return random.choice(words)
    
    return None


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Sign Language Recognition API is running"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint pour la reconnaissance en temps réel
    
    Reçoit: {"type": "frame", "data": "base64...", "timestamp": 123456}
    Envoie: {"type": "word", "word": "Hello"}
    """
    await websocket.accept()
    print("✅ Client connecté")
    
    frame_count = 0
    
    try:
        while True:
            # Recevoir le message du frontend
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "frame":
                frame_count += 1
                frame_data = message.get("data")
                timestamp = message.get("timestamp")
                
                # Log toutes les 25 frames (~1 seconde à 25 FPS)
                if frame_count % 25 == 0:
                    print(f"📹 Frames reçues: {frame_count}")
                
                # Reconnaissance du signe
                word = recognize_sign(frame_data)
                
                # Renvoyer le mot si détecté
                if word:
                    print(f"🗣️ Mot détecté: {word}")
                    await websocket.send_text(json.dumps({
                        "type": "word",
                        "word": word
                    }))
                    
    except WebSocketDisconnect:
        print(f"❌ Client déconnecté (total frames: {frame_count})")
    except Exception as e:
        print(f"❌ Erreur: {e}")


# Point d'entrée alternatif sur la racine WebSocket (pour compatibilité)
@app.websocket("/")
async def websocket_root(websocket: WebSocket):
    """WebSocket sur la racine (redirection vers /ws)"""
    await websocket_endpoint(websocket)


if __name__ == "__main__":
    import uvicorn
    print("🚀 Démarrage du serveur...")
    print("📡 WebSocket disponible sur: ws://localhost:8000/ws")
    print("💡 Utilise ngrok pour exposer: ngrok http 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

