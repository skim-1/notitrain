from flask import Flask, request, jsonify
from flask_cors import cross_origin
import json

app = Flask(__name__)

def import_json():
    with open('./recipes.json') as f:
        return json.load(f)

def dump_json(injson):
    with open('./recipes.json', 'w') as f:
        json.dump(injson, f)


@app.route('/clear', methods=['POST'])
@cross_origin()
def clear():
    if request.method == 'POST':
        f=request.json
        if f['key'] == ";F05lUCw%[hUeD?84~XMK{E@OO}4uPdYB4do'?8bH-BlSTuh9#|W=TzS(Pq9e":
            dump_json({'recipes': []})
            out = jsonify(msg='done')
            return out
        else:
            out = jsonify(msg='invalid key')
            return out


