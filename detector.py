import cv2
import math
import numpy as np
from bottle import abort
from logging import getLogger
import numpy as np

logger = getLogger(__name__)


classes = {
    "m1": 0, "m2": 1, "m3": 2, "m4": 3, "m5": 4, "m6": 5, "m7": 6, "m8": 7, "m9": 8,
    "p1": 9, "p2": 10, "p3": 11, "p4": 12, "p5": 13, "p6": 14, "p7": 15, "p8": 16, "p9": 17,
    "s1": 18, "s2": 19, "s3": 20, "s4": 21, "s5": 22, "s6": 23, "s7": 24, "s8": 25, "s9": 26,
    "P": 27, "F": 28, "C": 29, "E": 30, "S": 31, "W": 32, "N": 33
}

'''
json sample
[{'label': 'p3', 'bottomright': {'y': 623, 'x': 1065}, 'topleft': {'y': 536, 'x': 951}, 'confidence': 0.77076006},
 {'label': 'p3', 'bottomright': {'y': 666, 'x': 959}, 'topleft': {'y': 552, 'x': 861}, 'confidence': 0.53740776},
 {'label': 'P', 'bottomright': {'y': 410, 'x': 463}, 'topleft': {'y': 306, 'x': 384}, 'confidence': 0.86410844},
 {'label': 'P', 'bottomright': {'y': 405, 'x': 539}, 'topleft': {'y': 300, 'x': 457}, 'confidence': 0.8059135},
 {'label': 'P', 'bottomright': {'y': 407, 'x': 609}, 'topleft': {'y': 304, 'x': 526}, 'confidence': 0.857607},
 {'label': 'F', 'bottomright': {'y': 404, 'x': 684}, 'topleft': {'y': 308, 'x': 605}, 'confidence': 0.8254916},
 {'label': 'F', 'bottomright': {'y': 409, 'x': 764}, 'topleft': {'y': 315, 'x': 668}, 'confidence': 0.7356734},
 {'label': 'F', 'bottomright': {'y': 407, 'x': 837}, 'topleft': {'y': 317, 'x': 732}, 'confidence': 0.8032853},
 {'label': 'C', 'bottomright': {'y': 418, 'x': 923}, 'topleft': {'y': 314, 'x': 833}, 'confidence': 0.8658796},
 {'label': 'C', 'bottomright': {'y': 436, 'x': 1013}, 'topleft': {'y': 327, 'x': 907}, 'confidence': 0.8479251},
 {'label': 'C', 'bottomright': {'y': 429, 'x': 1055}, 'topleft': {'y': 325, 'x': 967}, 'confidence': 0.8740596},
 {'label': 'S', 'bottomright': {'y': 393, 'x': 96}, 'topleft': {'y': 292, 'x': 4}, 'confidence': 0.8791591},
 {'label': 'S', 'bottomright': {'y': 388, 'x': 166}, 'topleft': {'y': 285, 'x': 74}, 'confidence': 0.7881428},
 {'label': 'S', 'bottomright': {'y': 383, 'x': 244}, 'topleft': {'y': 287, 'x': 161}, 'confidence': 0.8698105},
 {'label': 'W', 'bottomright': {'y': 390, 'x': 314}, 'topleft': {'y': 287, 'x': 231}, 'confidence': 0.8952105},
 {'label': 'W', 'bottomright': {'y': 408, 'x': 396}, 'topleft': {'y': 303, 'x': 313}, 'confidence': 0.9040169}] 
'''

IMAGE_SIZE = 512

def detect_tiles(tfnet, path) -> (bool, dict):
    img = cv2.imread(path)
    # ローテーションが必要なら入れる
    img_h, img_w = calc_resized_hw(*img.shape[:2])
    img = add_padding(img)
    logger.debug("img h: {}, w:{}".format(img_h, img_w))
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    cv2.imwrite("./images/tmp.jpg", img)
    result = tfnet.return_predict(img)
    return flow2hand(result, img_w, img_h)

def calc_resized_hw(img_h, img_w) -> (int, int):
    h = IMAGE_SIZE if img_h >= img_w else int(IMAGE_SIZE * img_h / img_w)
    w = IMAGE_SIZE if img_w >= img_h else int(IMAGE_SIZE * img_w / img_h)
    return h, w

def add_padding(img) -> np.ndarray:
    '''
    padding処理
    適当に端から合わせればいいと思う。
    '''
    h,w = img.shape[:2]
    size = max(h, w)
    padded = np.zeros((size, size, 3))
    padded[0:h, 0:w, :] = img[:]
    return padded

def flow2hand(result, x, y) -> (bool, dict):
    agari = [tile["label"] for tile in result
        if tile["bottomright"]["y"] / y >= 0.5
        and tile["topleft"]["x"] / x <= 0.2]
    backs = [tile for tile in result
        if tile["label"] == "back"]
    opened_tiles = [tile for tile in result
        if tile["label"] != "back"
        and tile["bottomright"]["y"] / y < 0.5]
    # 面前部分はこれでOK
    hidden_hands = [tile["label"] for tile in result
        if tile["bottomright"]["y"] / y >= 0.5
        and tile["bottomright"]["x"] / x > 0.2]
    logger.debug("agari: {}".format(agari))
    logger.debug("backs: {}".format(backs))
    logger.debug("opened_tiles: {}".format(opened_tiles))
    logger.debug("hidden_tiles: {}".format(hidden_hands))
    if len(agari) != 1:
        logger.info("和了牌の枚数が不正です（{} 枚）".format(len(agari)))
        return detection_fail()
    if len(backs) % 2 != 0:
        logger.info("背面が映っている牌の枚数が不正です（{} 枚）".format(len(backs)))
        return detection_fail()
    # 暗槓
    back_count = {}
    opened_tiles.sort(key=lambda t: t["bottomright"]["x"])
    checked = [False for i in range(len(opened_tiles))]
    for back in backs:
        min_dist = int(1e+9)
        idx = -1
        for i, tile in enumerate(opened_tiles):
            dist = calc_dist(back, tile)
            if not checked[i] and dist < min_dist:
                min_dist = dist
                idx = i
        logger.debug("idx: {}".format(idx))
        checked[idx] = True
        if opened_tiles[idx]["label"] in back_count:
            back_count[opened_tiles[idx]["label"]] += 1
        else:
            back_count[opened_tiles[idx]["label"]] = 1
    hidden_kans = []
    for k, v in back_count.items():
        if v == 2:
            hidden_kans.append(k) 
            continue
        # 牌の裏面が一枚しか認識されていない場合
        # →openedに同じ牌が入っていればそれを使うことにする
        for i, tile in enumerate(opened_tiles):
            if not checked[i] and tile["label"] == k:
                hidden_kans.append(k)
                checked[i] = True
                break
            if i == len(opened_tiles) - 1:
                logger.info("暗槓が正しく認識できていません")
                logger.debug("back_count: {}".format(back_count))
                logger.debug("now: {k}".format(back_count, k))
                return detection_fail()

    # 座標を左からソートして貪欲に鳴きの組を作成する。
    # 同じ程度のy座標には高々2組しか存在しないと仮定している。
    sets = []
    for i, tile in enumerate(opened_tiles):
        if checked[i]:
            continue 
        pair_candidates = [(t,j) for j, t in enumerate(opened_tiles)
                            if not checked[j] and similar_height(tile, t)]
        for tile,j in pair_candidates:
            checked[j] = True
        if len(pair_candidates) in [3, 4]:
            sets.append([t["label"] for t, j in pair_candidates])
        elif len(pair_candidates) > 6 and pair_candidates[2][0]["label"] == pair_candidates[3][0]["label"]:
            sets.append([t["label"] for t, j in pair_candidates[:4]])
            sets.append([t["label"] for t, j in pair_candidates[4:]])
        else:
            sets.append([t["label"] for t, j in pair_candidates[:3]])
            sets.append([t["label"] for t, j in pair_candidates[3:]])
    opened_kans = []
    opened_hands = []
    logger.debug(sets)
    for p in sets:
        if len(p) == 3:
            opened_hands.append(p[:])
        elif len(p) == 4:
            opened_kans.append(p[0])
        else:
            detection_fail()
    response = {
        'agari': agari[0],
        'opened': opened_hands,
        'hidden': hidden_hands,
        "kan" : {
            'opened': opened_kans,
            'hidden': hidden_kans
        }
    }
    logger.debug(response)
    return True, response
            
HEIGHT_RANGE = 30
def similar_height(tile1, tile2):
    dy = abs((tile1["bottomright"]["y"] + tile1["topleft"]["y"]) / 2
     - (tile2["bottomright"]["y"] + tile2["topleft"]["y"]) / 2)
    return dy <= HEIGHT_RANGE

def seems_near(tile1, tile2):
    dy = abs((tile1["bottomright"]["y"] + tile1["topleft"]["y"]) / 2
     - (tile2["bottomright"]["y"] + tile2["topleft"]["y"]) / 2)
    dx = min([abs(tile1["bottomright"]["x"] - tile2["topleft"]["x"]),
                abs(tile2["bottomright"]["x"] - tile1["topleft"]["x"])])
    # 適当
    return dy < 20 and dx < 20
    
def seems_pair(tile1, tile2):
    return seems_near(tile1, tile2) and seems_pairable(tile1, tile2)

def seems_pairable(tile1, tile2):
    # ガバい
    t1_label = tile1["label"]
    t2_label = tile2["label"]
    return (is_vals[t1_label] and t1_label == t2_label
        or (is_ms[t2_label] and is_ms[t2_label]
        or is_ps[t1_label] and is_ps[t2_label]
        or is_ss[t1_label] and is_ss[t2_label])
        and abs(classes[t1_label] - classes[t2_label]) <= 2)

def is_ms(tile_str):
    return classes[tile_str] < 9

def is_ps(tile_str):
    return (classes[tile_str] >= 9
            and classes[tile_str] < 18)
            
def is_ss(tile_str):
    return (classes[tile_str] >= 18 
            and classes[tile_str] < 27)
    
def is_vals(tile_str):
    return classes[tile_str] >= 27

def calc_dist(tile1, tile2):
    # x: 左右端
    # y: 重心
    dy = abs((tile1["bottomright"]["y"] + tile1["topleft"]["y"]) / 2
     - (tile2["bottomright"]["y"] + tile2["topleft"]["y"]) / 2)
    dx = min([abs(tile1["bottomright"]["x"] - tile2["topleft"]["x"]),
                abs(tile2["bottomright"]["x"] - tile1["topleft"]["x"])])
    return math.sqrt(dy**2 + dx**2)

def detection_fail():
    return False, {}
    # abort(400, "Detection failed.")

