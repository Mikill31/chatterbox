import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("Cargando el modelo...")

modelo = ChatterboxMultilingualTTS.from_pretrained(
    device="cpu"
)

texto="Bonjour les petites étoiles, la Terre vous dit bonjouuur !"

print("Generando el audio...")

audio = modelo.generate(
    texto,
    language_id="fr",
    exaggeration=0.7, # expresividad o emoción
    cfg_weight=0.4, #Afecta velocidad y ritmo
    temperature=0.8, #Controla la variabilidad.
    repetition_penalty=1.2, #Penaliza repetciones extrañas
    min_p=0.05, # Filtra opciones improbables durante la generacion
    top_p=1.0 #Controla el muestreo probabilistico  
)

ta.save(
    "audio.wav",
    audio,
    modelo.sr
)

print("Audio generado.")

