if __name__=="__main__":
    params = {'gamma': [0.0001],
              'cw': [{0: 1, 1: 1}],
              'C': [10.0],
              'kernel': ['sigmoid']}
    for cw in params['cw']:
        print(cw)