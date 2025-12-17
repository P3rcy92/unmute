# 🤟 Backend - Sign Language Recognition

Backend WebSocket pour la reconnaissance de langue des signes en temps réel.

## 📋 Prérequis

- Python 3.10+
- ngrok (pour exposer le serveur)

## 🚀 Installation

### 1. Créer un environnement virtuel

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer le serveur

```bash
python main.py
```

Le serveur démarre sur `http://localhost:8000`

### 4. Exposer avec ngrok

Dans un autre terminal :

```bash
ngrok http 8000
```

Tu obtiendras une URL du type : `https://abc123.ngrok.io`

**⚠️ Important** : Pour le frontend, utilise `wss://abc123.ngrok.io/ws` (avec `wss://` et `/ws`)

## 📡 Contrat WebSocket

### Endpoint

```
ws://localhost:8000/ws
# ou via ngrok:
wss://abc123.ngrok.io/ws
```

### Messages reçus (du frontend)

Le frontend envoie **25 frames par seconde** :

```json
{
  "type": "frame",
  "data": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "timestamp": 1702834567890
}
```

- `type`: Toujours "frame"
- `data`: Image JPEG encodée en base64 (640x480, qualité 70%)
- `timestamp`: Timestamp Unix en millisecondes

### Messages envoyés (vers le frontend)

Envoie le mot reconnu à **chaque frame** (le frontend gère la déduplication) :

```json
{
  "type": "word",
  "word": "Hello"
}
```

- `type`: Toujours "word"
- `word`: Le mot reconnu par le modèle ML

**Note importante** : Le frontend filtre automatiquement les mots répétés. Tu peux donc envoyer le même mot plusieurs fois, seul le premier sera traité et envoyé au TTS.

## 🧠 Intégration du modèle ML

Modifie la fonction `recognize_sign()` dans `main.py` :

```python
def recognize_sign(frame_data: str) -> str | None:
    # 1. Extraire l'image base64
    base64_string = frame_data.split(",")[1]
    image_bytes = base64.b64decode(base64_string)
    
    # 2. Convertir en format utilisable (exemple avec OpenCV)
    import numpy as np
    import cv2
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 3. Passer au modèle ML
    word = ton_modele.predict(image)
    
    return word  # ou None si rien détecté
```

## 📁 Structure

```
backend/
├── main.py           # Serveur FastAPI + WebSocket
├── requirements.txt  # Dépendances Python
└── README.md         # Ce fichier
```

## 🔧 Configuration frontend

Le frontend doit configurer cette variable d'environnement :

```bash
# .env.local (côté frontend Next.js)
NEXT_PUBLIC_WS_URL=wss://abc123.ngrok.io
```

## 📊 Spécifications techniques

| Paramètre | Valeur |
|-----------|--------|
| Frame rate | 25 FPS |
| Résolution | 640x480 |
| Format image | JPEG base64 |
| Qualité JPEG | 70% |
| Intervalle | 40ms |

## 🧪 Test rapide

1. Lance le serveur : `python main.py`
2. Ouvre `http://localhost:8000` → doit afficher `{"status": "ok", ...}`
3. Le mode mock renvoie des mots aléatoires pour tester la connexion

## 🔄 Flux complet

```
┌─────────────┐      25 FPS       ┌─────────────┐
│  FRONTEND   │ ── frames ──────► │   BACKEND   │
│  (Next.js)  │                   │  (FastAPI)  │
│             │ ◄── mots ──────── │   + ML      │
└──────┬──────┘                   └─────────────┘
       │
       │ mot unique (filtré)
       ▼
┌─────────────┐
│ ELEVENLABS  │
│   (TTS)     │
│      🔊     │
└─────────────┘
```

## 📝 Notes

- Le frontend filtre les mots répétés → tu peux renvoyer le même mot à chaque frame
- 25 frames/seconde = ton modèle doit traiter une frame en < 40ms idéalement
- En production, remplace `allow_origins=["*"]` par l'URL exacte du frontend
