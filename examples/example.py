from src.tvaegan_synthesizer import TVAEGANSynthesizer


synthesizer = TVAEGANSynthesizer()
synthesizer.fit(df_real)

df_synt = synthesizer.predict(size)
