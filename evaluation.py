from sklearn.metrics import f1_score

def f1_scoreMultilabs(y_pred,y_true):
    ''' 
    y_pred : batch*num_labs
    y_true : batch*num_labs
    '''
    f1= 0
    #print("pred.shape: "+str(y_pred))
    #print("true.shape: "+str(y_true.shape))
    for cur in range(len(y_pred)):
        f1+=f1_score(y_true, y_pred , average='macro')
        
    f1=f1/len(y_pred)
    return f1