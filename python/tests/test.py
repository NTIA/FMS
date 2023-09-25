import os
import pickle
import unittest

import numpy as np

import fms


class TestFMS(unittest.TestCase):
    """
    Tests fixed-size modulation spectra (FMS) code from FMS package.
    """

    dirname = os.path.dirname(__file__)
    reference_folder = os.path.join(dirname, "../reference_files")
    wav_path = os.path.join(dirname, "../../wavs")

    fnames = [
        "audioShort16k.wav",  #     %fs = 16k,      length appx. 1.6 sec
        "audioLong16k.wav",  #      %fs = 16k,      length appx. 3.5 sec
        "audio22k.wav",  #          %fs = 22.05k,   length appx. 3.3 sec
        "audio32k.wav",  #          %fs = 32k,      length appx. 2.9 sec
        "audio44k.wav",  #          %fs = 44.1k,    length appx. 5.8 sec
        "audio48k.wav",  #          %fs = 48k,      length appx. 3.0 sec
    ]

    def test_fms(self):
        for fname in self.fnames:
            filepath = os.path.join(self.wav_path, fname)
            # Load audio
            fs, audio = fms.load_audio(filepath)
            # Get FMS representation
            PsiM, PsiP = fms.fms(audio, fs)
            rep = dict(magnitude=PsiM, phase=PsiP)

            name = os.path.basename(filepath)
            name, _ = os.path.splitext(name)

            ref_path = os.path.join(self.reference_folder, name + "Ref.pkl")
            with open(ref_path, "rb") as f:
                reference = pickle.load(f)

            for key in ["magnitude", "phase"]:
                with self.subTest(filename=fname, value=key):
                    abserror = np.mean(np.abs(rep[key] - reference[key]))
                    self.assertEqual(abserror, 0)


if __name__ == "__main__":
    unittest.main()
