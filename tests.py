import unittest
from detector import flow2hand, calc_resized_hw
from logging import basicConfig, getLogger, DEBUG


class TestFlow2Hand(unittest.TestCase):
    def test_hidden_kan(self):
        result = [{'label': 'm5', 'topleft': {'y': 195, 'x': 192}, 'confidence': 0.9307989, 'bottomright': {'y': 262, 'x': 244}}, {'label': 'm6', 'topleft': {'y': 194, 'x': 238}, 'confidence': 0.9055795, 'bottomright': {'y': 261, 'x': 287}}, {'label': 'm7', 'topleft': {'y': 197, 'x': 285}, 'confidence': 0.90744555, 'bottomright': {'y': 260, 'x': 334}}, {'label': 'p1', 'topleft': {'y': 296, 'x': 196}, 'confidence': 0.8782565, 'bottomright': {'y': 359, 'x': 244}}, {'label': 'p2', 'topleft': {'y': 297, 'x': 242}, 'confidence': 0.85990447, 'bottomright': {'y': 365, 'x': 295}}, {'label': 'p3', 'topleft': {'y': 204, 'x': 29}, 'confidence': 0.9225853, 'bottomright': {'y': 272, 'x': 90}}, {'label': 'p5', 'topleft': {'y': 198, 'x': 328}, 'confidence': 0.8885702, 'bottomright': {'y': 258, 'x': 385}}, {'label': 'p5', 'topleft': {'y': 203, 'x': 378}, 'confidence': 0.78256285, 'bottomright': {'y': 260, 'x': 428}}, {'label': 'p5', 'topleft': {'y': 208, 'x': 400}, 'confidence': 0.79103637, 'bottomright': {'y': 273, 'x': 472}}, {'label': 's7', 'topleft': {'y': 307, 'x': 340}, 'confidence': 0.7892322, 'bottomright': {'y': 370, 'x': 397}}, {'label': 's7', 'topleft': {'y': 308, 'x': 372}, 'confidence': 0.83537805, 'bottomright': {'y': 371, 'x': 431}}, {'label': 'W', 'topleft': {'y': 66, 'x': 166}, 'confidence': 0.83687836, 'bottomright': {'y': 121, 'x': 210}}, {'label': 'W', 'topleft': {'y': 72, 'x': 301}, 'confidence': 0.863754, 'bottomright': {'y': 131, 'x': 341}}, {'label': 'back', 'topleft': {'y': 65, 'x': 209}, 'confidence': 0.7821499, 'bottomright': {'y': 120, 'x': 266}}, {'label': 'back', 'topleft': {'y': 69, 'x': 243}, 'confidence': 0.7050225, 'bottomright': {'y': 132, 'x': 301}}]
        ret = flow2hand(result, 512, 384)
        self.assertTrue(ret[0])
        self.assertEqual(len(ret[1]["hidden_hands"]), 10)
        self.assertEqual(ret[1]["agari"], "p3")
        self.assertEqual(ret[1]["hidden_kans"], ["W"])

    def test_saki1(self):
        result = [{'label': 'p1', 'topleft': {'y': 31, 'x': 277}, 'confidence': 0.8881509, 'bottomright': {'y': 78, 'x': 340}}, {'label': 'p1', 'topleft': {'y': 18, 'x': 343}, 'confidence': 0.79679304, 'bottomright': {'y': 72, 'x': 394}}, {'label': 'p1', 'topleft': {'y': 20, 'x': 371}, 'confidence': 0.5958569, 'bottomright': {'y': 78, 'x': 434}}, {'label': 'p1', 'topleft': {'y': 19, 'x': 429}, 'confidence': 0.69431233, 'bottomright': {'y': 72, 'x': 475}}, {'label': 'p2', 'topleft': {'y': 14, 'x': 48}, 'confidence': 0.8728946, 'bottomright': {'y': 80, 'x': 104}}, {'label': 'p2', 'topleft': {'y': 13, 'x': 187}, 'confidence': 0.88905394, 'bottomright': {'y': 78, 'x': 241}}, {'label': 'p3', 'topleft': {'y': 105, 'x': 169}, 'confidence': 0.8979264, 'bottomright': {'y': 177, 'x': 227}}, {'label': 'p3', 'topleft': {'y': 104, 'x': 316}, 'confidence': 0.85143423, 'bottomright': {'y': 173, 'x': 376}}, {'label': 'p4', 'topleft': {'y': 251, 'x': 277}, 'confidence': 0.77447337, 'bottomright': {'y': 317, 'x': 335}}, {'label': 'p4', 'topleft': {'y': 248, 'x': 338}, 'confidence': 0.7007526, 'bottomright': {'y': 315, 'x': 393}}, {'label': 'p4', 'topleft': {'y': 241, 'x': 373}, 'confidence': 0.8776228, 'bottomright': {'y': 313, 'x': 439}}, {'label': 'p5', 'topleft': {'y': 248, 'x': 26}, 'confidence': 0.9146055, 'bottomright': {'y': 329, 'x': 95}}, {'label': 'p5', 'topleft': {'y': 253, 'x': 219}, 'confidence': 0.82119465, 'bottomright': {'y': 327, 'x': 277}}, {'label': 'back', 'topleft': {'y': 17, 'x': 84}, 'confidence': 0.5023014, 'bottomright': {'y': 78, 'x': 143}}, {'label': 'back', 'topleft': {'y': 14, 'x': 142}, 'confidence': 0.80395395, 'bottomright': {'y': 74, 'x': 196}}, {'label': 'back', 'topleft': {'y': 104, 'x': 226}, 'confidence': 0.8182925, 'bottomright': {'y': 175, 'x': 274}}, {'label': 'back', 'topleft': {'y': 104, 'x': 272}, 'confidence': 0.7305131, 'bottomright': {'y': 171, 'x': 328}}]
        ret = flow2hand(result, 512, 384)
        self.assertTrue(ret[0])
        self.assertEqual(ret[1]["opened_kans"], ["p1"])
        self.assertEqual(set(ret[1]["hidden_kans"]), set(["p2", "p3"]))

    def test_s_9ren(self):
        result = [{'confidence': 0.89377123, 'bottomright': {'x': 271, 'y': 263}, 'label': 's1', 'topleft': {'x': 221, 'y': 202}}, {'confidence': 0.73918736, 'bottomright': {'x': 365, 'y': 263}, 'label': 's1', 'topleft': {'x': 313, 'y': 205}}, {'confidence': 0.7109291, 'bottomright': {'x': 402, 'y': 268}, 'label': 's1', 'topleft': {'x': 338, 'y': 203}}, {'confidence': 0.93885094, 'bottomright': {'x': 92, 'y': 300}, 'label': 's1', 'topleft': {'x': 30, 'y': 224}}, {'confidence': 0.83469963, 'bottomright': {'x': 352, 'y': 357}, 'label': 's2', 'topleft': {'x': 299, 'y': 293}}, {'confidence': 0.89779395, 'bottomright': {'x': 253, 'y': 349}, 'label': 's3', 'topleft': {'x': 200, 'y': 287}}, {'confidence': 0.87354594, 'bottomright': {'x': 298, 'y': 347}, 'label': 's4', 'topleft': {'x': 254, 'y': 288}}, {'confidence': 0.9076421, 'bottomright': {'x': 205, 'y': 355}, 'label': 's5', 'topleft': {'x': 148, 'y': 287}}, {'confidence': 0.79107445, 'bottomright': {'x': 390, 'y': 358}, 'label': 's6', 'topleft': {'x': 341, 'y': 293}}, {'confidence': 0.70499766, 'bottomright': {'x': 440, 'y': 359}, 'label': 's7', 'topleft': {'x': 382, 'y': 294}}, {'confidence': 0.8631175, 'bottomright': {'x': 486, 'y': 362}, 'label': 's8', 'topleft': {'x': 432, 'y': 295}}, {'confidence': 0.8970947, 'bottomright': {'x': 230, 'y': 264}, 'label': 's9', 'topleft': {'x': 174, 'y': 198}}, {'confidence': 0.8987541, 'bottomright': {'x': 316, 'y': 262}, 'label': 's9', 'topleft': {'x': 272, 'y': 202}}, {'confidence': 0.9092049, 'bottomright': {'x': 445, 'y': 264}, 'label': 's9', 'topleft': {'x': 401, 'y': 206}}]
        ret = flow2hand(result, 512, 384)
        self.assertTrue(ret[0])
        self.assertEqual(ret[1]["agari"], "s1")
        self.assertEqual(ret[1]["hidden_hands"], [
            "s1", "s1", "s1", "s2", "s3", "s4", "s5", "s6",
            "s7", "s8", "s9", "s9", "s9"
        ])

    def test_s_hon2(self):
        result = [{'confidence': 0.91495997, 'bottomright': {'x': 332, 'y': 116}, 'label': 's1', 'topleft': {'x': 278, 'y': 55}}, {'confidence': 0.7918326, 'bottomright': {'x': 386, 'y': 108}, 'label': 's1', 'topleft': {'x': 332, 'y': 65}}, {'confidence': 0.850623, 'bottomright': {'x': 433, 'y': 112}, 'label': 's1', 'topleft': {'x': 379, 'y': 57}}, {'confidence': 0.53998095, 'bottomright': {'x': 467, 'y': 116}, 'label': 's1', 'topleft': {'x': 401, 'y': 57}}, {'confidence': 0.8675447, 'bottomright': {'x': 226, 'y': 114}, 'label': 's2', 'topleft': {'x': 172, 'y': 53}}, {'confidence': 0.9244233, 'bottomright': {'x': 179, 'y': 113}, 'label': 's3', 'topleft': {'x': 116, 'y': 62}}, {'confidence': 0.8529427, 'bottomright': {'x': 261, 'y': 114}, 'label': 's4', 'topleft': {'x': 216, 'y': 51}}, {'confidence': 0.91232646, 'bottomright': {'x': 204, 'y': 294}, 'label': 's7', 'topleft': {'x': 144, 'y': 228}}, {'confidence': 0.9323895, 'bottomright': {'x': 97, 'y': 290}, 'label': 's8', 'topleft': {'x': 36, 'y': 212}}, {'confidence': 0.8713463, 'bottomright': {'x': 247, 'y': 293}, 'label': 's9', 'topleft': {'x': 196, 'y': 228}}, {'confidence': 0.8626074, 'bottomright': {'x': 296, 'y': 291}, 'label': 's9', 'topleft': {'x': 244, 'y': 225}}, {'confidence': 0.9003637, 'bottomright': {'x': 341, 'y': 291}, 'label': 's9', 'topleft': {'x': 289, 'y': 224}}, {'confidence': 0.8503369, 'bottomright': {'x': 389, 'y': 279}, 'label': 'F', 'topleft': {'x': 341, 'y': 227}}, {'confidence': 0.77307737, 'bottomright': {'x': 474, 'y': 281}, 'label': 'F', 'topleft': {'x': 430, 'y': 225}}, {'confidence': 0.7275932, 'bottomright': {'x': 432, 'y': 291}, 'label': 'F', 'topleft': {'x': 373, 'y': 227}}]
        ret = flow2hand(result, 512, 384)
        self.assertTrue(ret[0])
        self.assertEqual(ret[1]["opened_kans"], ["s1"])
        self.assertEqual(ret[1]["opened_hands"], [["s3", "s2", "s4"]])

class TestUtilMethods(unittest.TestCase):
    def test_calc_resized_hw(self):
        self.assertTrue(calc_resized_hw(512, 328),
            (512, 328))


if __name__ == "__main__":
    logger = getLogger(__name__)
    basicConfig(level=DEBUG)
    unittest.main()