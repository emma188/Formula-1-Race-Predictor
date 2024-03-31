import json

import pandas as pd

import hw593backend as back
from flask import Flask, render_template, request, redirect, jsonify

app = Flask(__name__)

classification = {
    "Neural Network": [['activation', 'identity'], ['hidden_layer_sizes', '(80, 20, 40, 5)'], ['solver', 'lbfgs'],
                       ['alpha', '0.1082636733874054']],
    "Random Forest": [['criterion', 'entropy'], ['max_features', 'auto'], ['max_depth', '49']],
    "SVM": [['kernel', 'sigmoid'], ['gamma', '0.0001'], ['C', '10.0'], ['cw', '(1, 1)']]}
regression = {"Linear Regressor": [],
              "SVM": [['kernel', 'sigmoid'], ['gamma', '0.0001'], ['C', '10.0'], ['cw', '(1, 1)']],
              "Neural Network": [['activation', 'identity'], ['hidden_layer_sizes', '(75, 30, 50, 10, 3)'],
                                 ['solver', 'sgd'], ['alpha', '0.4832930238571752']]}

model = model_score = None  # Don't touch this, this is where the trained model and their score is saved
instances = None
target_class = ""
target_model = ""
g_hyper = []
year = []
removed_feats = []
data_list = []


def getPrediction():
    global instances
    print(instances)
    t1,t2 = back.inferencemany(target_class.lower(),model,instances)
    return (t1,t2)

def createInstances():
    global instances,data_list
    for i in range(len(data_list)):
        instances.loc[i] = 0
        instances.loc[i]['weather_warm'] = False
        instances.loc[i]['weather_cold'] = False
        instances.loc[i]['weather_dry'] = False
        instances.loc[i]['weather_wet'] = False
        instances.loc[i]['weather_cloudy'] = False
        ob = data_list[i]
        for j in range(len(ob)):
            if(j==0):
                if('season' not in removed_feats):
                    if(isfloat(ob[j])):
                        instances.loc[i]['season'] = int(ob[j])
            elif(j==1):
                if ('round' not in removed_feats):
                    if (isfloat(ob[j])):
                        instances.loc[i]['round'] = int(ob[j])
            elif (j == 2):
                if ('circuit_id' not in removed_feats):
                    if(f'circuit_id_{ob[j]}' in instances.columns):
                        instances.loc[i][f'circuit_id_{ob[j]}'] = 1
            elif (j == 3):
                if ('weather' not in removed_feats):
                    instances.loc[i][f'weather_{ob[j]}'] = True
            elif (j == 4):
                if ('nationality' not in removed_feats):
                    if(f'nationality_{ob[j].capitalize()}' in instances.columns):
                        instances.loc[i][f'nationality_{ob[j].capitalize()}'] = 1
            elif (j == 5):
                if ('constructor' not in removed_feats):
                    if (f'constructor_{ob[j]}' in instances.columns):
                        instances.loc[i][f'constructor_{ob[j]}'] = 1
            elif (j == 6):
                if ('grid' not in removed_feats):
                    if (isfloat(ob[j])):
                        instances.loc[i]['grid'] = int(ob[j])
            elif (j == 7):
                if ('driver_points' not in removed_feats):
                    if (isfloat(ob[j])):
                        instances.loc[i]['driver_points'] = int(ob[j])
            elif (j == 8):
                if ('driver_wins' not in removed_feats):
                    if (isfloat(ob[j])):
                        instances.loc[i]['driver_wins'] = int(ob[j])
            elif (j == 9):
                if ('driver_standings_pos' not in removed_feats):
                    if (isfloat(ob[j])):
                        instances.loc[i]['driver_standings_pos'] = int(ob[j])
            elif (j == 10):
                if ('constructor_points' not in removed_feats):
                    if (isfloat(ob[j])):
                        instances.loc[i]['constructor_points'] = int(ob[j])
            elif (j == 11):
                if ('constructor_wins' not in removed_feats):
                    if (isfloat(ob[j])):
                        instances.loc[i]['constructor_wins'] = int(ob[j])
            elif (j == 12):
                if ('constructor_standings_pos' not in removed_feats):
                    if (isfloat(ob[j])):
                        instances.loc[i]['constructor_standings_pos'] = int(ob[j])
            elif (j == 13):
                if ('qualifying_time' not in removed_feats):
                    if (isfloat(ob[j])):
                        instances.loc[i]['qualifying_time'] = float(ob[j])
            elif (j == 14):
                if ('driver_age' not in removed_feats):
                    if (isfloat(ob[j])):
                        instances.loc[i]['driver_age'] = int(ob[j])


def procFeat(feats):
    global instances
    df = back.read_data()
    print(type(feats),len(feats),feats)
    if (len(feats) == 0):
        df, instances = back.process_data(df)
        return df
    else:
        if('weather' in feats):
            ind = feats.index('weather')
            del feats[ind]
            feats.append('weather_warm')
            feats.append('weather_cold')
            feats.append('weather_dry')
            feats.append('weather_wet')
        df = df.drop(feats, axis=1)
        df, instances = back.process_data(df)
        return df

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def runModel():
    global model, model_score, removed_feats
    typ = target_class.lower()
    modtype = target_model.lower()
    if('' in removed_feats):
        del removed_feats[0]
    df = procFeat(removed_feats)
    print(df)
    if ('classification' in typ):
        if ('neural' in modtype):
            if (len(year) == 0):
                model, model_score = back.NN(df, hidden_layer_sizes=[int(i) for i in
                                                                     g_hyper[1].replace('(', '').replace(')', '').split(
                                                                         ',')], activation=g_hyper[0].lower(),
                                             solver=g_hyper[2].lower(), alpha=float(g_hyper[3]))
            elif (len(year) == 2):
                model, model_score = back.NN(df=df, hidden_layer_sizes=[int(i) for i in
                                                                        g_hyper[1].replace('(', '').replace(')',
                                                                                                            '').split(
                                                                            ',')], activation=g_hyper[0].lower(),
                                             solver=g_hyper[2].lower(), alpha=float(g_hyper[3]), year=year[1],
                                             year2=year[0])
        elif ('forest' in modtype):
            if (isfloat(g_hyper[1])):
                maxf = float(g_hyper[1])
            else:
                maxf = g_hyper[1].lower()
            maxd = int(g_hyper[2])
            if (maxd == -1):
                maxd = None
            if (len(year) == 0):
                model, model_score = back.ranFor(df=df, crit=g_hyper[0].lower(), maxf=maxf, maxd=maxd)
            elif (len(year) == 2):
                model, model_score = back.ranFor(df=df, crit=g_hyper[0].lower(), maxf=maxf, maxd=maxd, year=year[1],
                                                 year2=year[0])
        elif ('svm' in modtype):
            if (isfloat(g_hyper[1])):
                gamma = float(g_hyper[1])
            else:
                gamma = g_hyper[1].lower()

            if (len(g_hyper) < 4):
                cw = {0: 1, 1: 1}
            else:
                t1 = g_hyper[3].replace('(', '').replace(')', '').split(',')
                cw = {0: t1[0], 1: t1[1]}
            if (len(year) == 0):
                model, model_score = back.svmmod(df=df, cw=cw, c=float(g_hyper[2]), kernel=g_hyper[0].lower(),
                                                 gamma=gamma)
            elif (len(year) == 2):
                model, model_score = back.svmmod(df=df, cw=cw, c=float(g_hyper[2]), kernel=g_hyper[0].lower(),
                                                 gamma=gamma, year=year[1],
                                                 year2=year[0])
    elif ('regression' in typ):
        if ('neural' in modtype):
            if (len(year) == 0):
                model, model_score = back.NNReg(df, hidden_layer_sizes=[int(i) for i in
                                                                        g_hyper[1].replace('(', '').replace(')',
                                                                                                            '').split(
                                                                            ',')], activation=g_hyper[0].lower(),
                                                solver=g_hyper[2].lower(), alpha=float(g_hyper[3]))
            elif (len(year) == 2):
                model, model_score = back.NNReg(df=df, hidden_layer_sizes=[int(i) for i in
                                                                           g_hyper[1].replace('(', '').replace(')',
                                                                                                               '').split(
                                                                               ',')], activation=g_hyper[0].lower(),
                                                solver=g_hyper[2].lower(), alpha=float(g_hyper[3]), year=year[1],
                                                year2=year[0])
        elif ('linear' in modtype):
            if (len(year) == 0):
                model, model_score = back.linreg(df=df)
            elif (len(year) == 2):
                model, model_score = back.linreg(df=df, year=year[1], year2=year[0])
        elif ('svm' in modtype):
            if (isfloat(g_hyper[1])):
                gamma = float(g_hyper[1])
            else:
                gamma = g_hyper[1].lower()
            if(len(g_hyper)<4):
                cw = {0: 1, 1: 1}
            else:
                t1 = g_hyper[3].replace('(', '').replace(')', '').split(',')
                cw = {0: t1[0], 1: t1[1]}
            if (len(year) == 0):
                model, model_score = back.svmreg(df=df, cw=cw, c=float(g_hyper[2]), kernel=g_hyper[0].lower(),
                                                 gamma=gamma)
            elif (len(year) == 2):
                model, model_score = back.svmreg(df=df, cw=cw, c=float(g_hyper[2]), kernel=g_hyper[0].lower(),
                                                 gamma=gamma, year=year[1],
                                                 year2=year[0])


@app.route('/')
def hello():
    return redirect("/bar")


@app.route('/index')
def index():
    return render_template('proj1.html')


@app.route('/bar')
def bar():
    return render_template('bar.html')


# 柱子的一些参数
@app.route('/bar_data')
def bar_data():
    data1 = [51.23, 53.45, 55.46, "-", "-", "-"]
    data2 = ["-", "-", "-", 58.6, 55.03, 56.19]
    data3 = ["Linear", "SVM", "Neural Network", "Neural Network", "Random Forest", "SVM"]
    return [data1, data2, data3]


# class+model_name
@app.route('/tooltitle')
def tooltitle():
    # 传回来的
    index = request.args.get("cata_model")
    print(index)
    model_name = index.split(',')
    hyperparameters = []
    models = []
    if model_name[0] == "classification":
        models = classification.keys()
        for m in models:
            if m == model_name[1]:
                hyperparameters = classification[m]
    else:
        models = regression.keys()
        for m in models:
            if m == model_name[1]:
                hyperparameters = regression[m]

    print(hyperparameters)
    hyper = {}

    for i in range(len(hyperparameters)):
        hyper[hyperparameters[i][0]] = hyperparameters[i][1]
    print(hyper)

    # data = {"A":12,"B":2,"12":2}#tuple((("A",12),("B",2),("C",3)))
    return jsonify(hyper)


@app.route('/index', methods=['POST'])
def my_form_post():
    global target_class, target_model, g_hyper, year, removed_feats
    g_hyper = []
    year = []
    removed_feats = []
    data_list = []
    text = request.form
    print(text)

    removed_feats = json.loads(text['text1'])
    model = json.loads(text['text2'])
    # text3 = json.loads(text['text3'])
    years = json.loads(text['text4'])
    hyper = json.loads(text['oper'])

    target_class = model['type']
    target_model = model['model']
    for i in range(len(years)):
        years[i] = int(years[i])
    year = years
    if hyper:
        g_hyper = list(hyper.values())
    else:
        if target_class == "classification":
            para_list = classification[target_model]
            for i in range(len(para_list)):
                g_hyper.append(para_list[i][1])
        else:
            para_list = regression[target_model]
            for i in range(len(para_list)):
                g_hyper.append(para_list[i][1])

    print(g_hyper,type(g_hyper))
    print(type(hyper),hyper)
    runModel()
    return render_template('page3.html')


@app.route("/prediction", methods=['GET', "POST"])
def my_prediction():
    global data_list,instances
    data = json.loads(request.form.get("data"))
    print(data)
    data_list = []
    print('Data list:', data_list)
    instances = instances[0:0]
    print('Data list 2:', instances)
    for i in range(len(data)):
        value = data[i].values()
        values = list(value)
        # print(values)
        weather = list(data[i].keys())[len(values) - 1]
        values.insert(3, weather)
        values.pop()
        data_list.append(values)
    print(data_list)
    # for i in range(len(data)):
    #     print(values)
    createInstances()
    winner, winchance = getPrediction()
    print("Prediction:", winner, winchance)
    list_x = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    list_y = ['season', 'round', 'circuit_id', 'weather',
              'nationality', 'constructor', 'grid', 'driver_points', 'driver_wins', 'driver_standings_pos',
              'constructor_points',
              'constructor_wins', 'constructor_standings_pos', 'qualifying_time', 'driver_age']
    ret = "Driver "+str(winner+1)+" have "+ str(winchance)+"% chance to win"
    return {"data": [ret], "list_x": list_x, "list_y": list_y}


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    # app.run()
    app.run(host='127.0.0.1', port=5005, debug=True)
