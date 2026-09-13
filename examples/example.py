import logging

from tvaegan.tvaegan_synthesizer import TVAEGANSynthesizer
import pandas as pd

logging.basicConfig(level=logging.INFO)

df_real = pd.read_csv('adult.csv')
synthesizer = TVAEGANSynthesizer(epochs=1)
synthesizer.fit(df_real)

df_synt = synthesizer.predict(len(df_real))
df_synt.to_csv('adult_synt.csv', index=False)
