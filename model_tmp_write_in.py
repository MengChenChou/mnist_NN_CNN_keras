import tensorflow as tf
import json
def aa():
    return tf.keras.models.load_model('C:/Users/User/PycharmProjects/ai_deep/mode_iv3LeafFinetune.h5')
import time
start = time.process_time()
with open('C:/Users/User/PycharmProjects/ai_deep/deeptest.json', 'r' ,encoding="big5") as outfile:
    target_buy_index_Dic = json.loads(outfile.read())
    outfile.close()
model = tf.contrib.keras.models.model_from_json(target_buy_index_Dic)
end = time.process_time()
print('推論圖片花費時間:', round(end - start), '秒')