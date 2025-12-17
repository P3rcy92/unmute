# Backend WebSocket pour la reconnaissance de langue des signes
# ============================================================

import json
import base64
import time
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
    
    # Mock pour tester la connexion (renvoie un mot aléatoire ~toutes les 2 secondes)
    import random
    if random.random() < 0.02:  # ~1 fois toutes les 50 frames (2 sec à 25 FPS)
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
    
    client_info = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
    print(f"\n{'='*50}")
    print(f"✅ Client connecté: {client_info}")
    print(f"{'='*50}\n")
    
    frame_count = 0
    words_sent = 0
    start_time = time.time()
    last_log_time = start_time
    
    try:
        while True:
            # Recevoir le message du frontend
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "frame":
                frame_count += 1
                frame_data = message.get("data")
                timestamp = message.get("timestamp")
                
                # Calculer la taille de l'image
                frame_size_kb = len(frame_data) / 1024 if frame_data else 0
                
                # Log toutes les secondes (25 frames à 25 FPS)
                current_time = time.time()
                if current_time - last_log_time >= 1.0:
                    elapsed = current_time - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"📥 Frames: {frame_count} | FPS réel: {fps:.1f} | Taille: {frame_size_kb:.1f}KB | Mots envoyés: {words_sent}")
                    last_log_time = current_time
                
                # Reconnaissance du signe
                word = recognize_sign(frame_data)
                
                # Renvoyer le mot si détecté
                if word:
                    words_sent += 1
                    print(f"📤 Mot envoyé: \"{word}\" (frame #{frame_count})")
                    await websocket.send_text(json.dumps({
                        "type": "word",
                        "word": word
                    }))
                    
    except WebSocketDisconnect:
        elapsed = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"❌ Client déconnecté: {client_info}")
        print(f"📊 Statistiques de session:")
        print(f"   - Durée: {elapsed:.1f}s")
        print(f"   - Frames reçues: {frame_count}")
        print(f"   - FPS moyen: {frame_count/elapsed:.1f}" if elapsed > 0 else "   - FPS moyen: N/A")
        print(f"   - Mots envoyés: {words_sent}")
        print(f"{'='*50}\n")
    except Exception as e:
        print(f"❌ Erreur: {e}")


# Point d'entrée alternatif sur la racine WebSocket (pour compatibilité)
@app.websocket("/")
async def websocket_root(websocket: WebSocket):
    """WebSocket sur la racine (redirection vers /ws)"""
    await websocket_endpoint(websocket)


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 Sign Language Recognition Backend")
    print("="*50)
    print("📡 WebSocket: ws://localhost:8000/ws")
    print("🌐 Health check: http://localhost:8000")
    print("💡 Expose with: ngrok http 8000")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
