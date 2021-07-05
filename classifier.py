
from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import accuracy_score

modelPack = {}

def trees( x_train, x_test, y_train, y_test ):
    res = []
    print("hello trees")
    m = tree.DecisionTreeClassifier()
    m.fit(x_train, y_train)
    print("fiting")
    predictions = m.predict(x_test)
    acc = accuracy_score(y_test,predictions)

    modelPack['DecisionTreeClassifier'] = m
    res.append( ( acc , "DecisionTreeClassifier" ) )

    m = tree.ExtraTreeClassifier()
    m.fit(x_train, y_train)
    predictions = m.predict(x_test)
    acc = accuracy_score(y_test,predictions)

    modelPack['ExtraTreeClassifier'] = m
    res.append( ( acc , "ExtraTreeClassifier" ) )

    print(res)
    return res

def ensembles( x_train, x_test, y_train, y_test ):
    res = []
    m = ensemble.BaggingClassifier()
    m.fit(x_train, y_train)
    predictions = m.predict(x_test)
    acc = accuracy_score(y_test,predictions)

    modelPack['BaggingClassifier'] = m
    res.append( ( acc , "BaggingClassifier" ) )
    m = ensemble.GradientBoostingClassifier()
    m.fit(x_train, y_train)
    predictions = m.predict(x_test)
    acc = accuracy_score(y_test,predictions)

    modelPack['GradientBoostingClassifier'] = m
    res.append( ( acc , "GradientBoostingClassifier" ) )
    return res


def classify( x_train, x_test, y_train, y_test ):
    result = {}
    r1 = trees( x_train, x_test, y_train, y_test )
    r2 = ensembles( x_train, x_test, y_train, y_test )
    res = r1 + r2
    res.sort(reverse=True)
    print(res)

    models = {}
    for val , name in res[:4]:
        result[name] = val
        models[name] = modelPack[name]
    return result , models