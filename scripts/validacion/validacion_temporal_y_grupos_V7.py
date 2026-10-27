#!/usr/bin/env python3
"""Validación complementaria V7 de ACCIDENTALIDAD.
Usa CSV reconstruido, produce resultados independientes de la V7 operativa.
Realiza tres particiones: aleatoria 80/20; por grupos 80/20; temporal 2025 -> 2026-02.
Modelos: media, Random Forest, HistGradientBoosting; MLP V7 opcional --mlp (requiere TensorFlow).
No modifica archivos originales ni sobrescribe salidas de la herramienta V7.
"""
import argparse, csv, math, json, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_poisson_deviance
from sklearn.dummy import DummyRegressor

CATS=['DPTO','MPIO','ACTIVEC']
NUM=['RELA_DEP','RELA_INDEP','AÑO_DE_INFORME','MES_DE_INFORME']
Y_COL='PRESUACCIDETRASUCE'
DEFAULT=Path('/mnt/data/Base_Positiva_2025_2026_RECONSTRUIDA_PRELIMINAR_NO_ENTRENAR.csv')

def load_csv(path):
    d={k:[] for k in CATS+NUM+[Y_COL]}
    with open(path,encoding='utf-8-sig',newline='') as f:
        rdr=csv.DictReader(f)
        for row in rdr:
            for k in CATS:d[k].append(row[k].strip())
            for k in NUM+[Y_COL]:
                val=row[k].strip().replace(',','')
                try: d[k].append(float(val))
                except ValueError: d[k].append(float('nan'))
    cats=np.array([d[k] for k in CATS],dtype=object).T
    nums0=np.array([d[k] for k in NUM],dtype=np.float64).T
    good=np.all(np.isfinite(nums0),axis=1)
    y=np.array(d[Y_COL],dtype=np.float64); y=np.maximum(np.nan_to_num(y,nan=0.0),0)
    # Equivale a regla de entrenamiento de V7
    workers=np.maximum(nums0[:,0]+nums0[:,1],1)
    numeric=np.column_stack([nums0, workers, np.log1p(workers)])
    periods=np.array([f'{int(year)}-{int(month):02d}' for year,month in zip(nums0[:,2],nums0[:,3])])
    group_names=['\x1f'.join(x) for x in cats]
    assert good.all(),f'{np.sum(~good)} filas inválidas requieren limpieza previa'
    return cats,numeric,y,periods,np.array(group_names,dtype=object)

def cap_lambda(y):
    p999=float(np.quantile(y,.999))
    c=max(1,2*p999,float(np.mean(y))+10*float(np.std(y)))
    c=min(c, float(np.max(y))*2 if np.max(y)>0 else c,500)
    return max(c,1.)

def metrics(yt,yp):
    yp=np.clip(np.array(yp,dtype=float),1e-8,None)
    return dict(MAE=mean_absolute_error(yt,yp),RMSE=np.sqrt(mean_squared_error(yt,yp)),
                R2=r2_score(yt,yp),desviacion_Poisson=mean_poisson_deviance(yt,yp),
                media_real=float(np.mean(yt)),media_pred=float(np.mean(yp)),
                reales_positivos=int(np.sum(yt>0)),reales_total=float(np.sum(yt)))

def mlp_pred(cat_train,cat_test,num_train,num_test,ytrain,cap):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    tf.keras.utils.set_random_seed(42)
    scale=StandardScaler().fit(num_train)
    xn=scale.transform(num_train).astype('float32');xt=scale.transform(num_test).astype('float32')
    enc=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    a=(enc.fit_transform(cat_train)+1).astype('int32')
    b=(enc.transform(cat_test)+1).astype('int32')
    inputs=[];em=[]
    ni=keras.Input(shape=(6,),name='num_input');inputs.append(ni)
    for j,name in enumerate(CATS):
        inp=keras.Input(shape=(1,),name=name+'_input'); inputs.append(inp)
        card=len(enc.categories_[j])+1
        dim=min(50,max(2,int(np.sqrt(card))))
        emb=layers.Embedding(input_dim=card+1,output_dim=dim,name='emb_'+name)(inp)
        em.append(layers.Flatten()(emb))
    x=layers.Concatenate()(em+[ni]); x=layers.Dense(128,activation='relu')(x)
    x=layers.Dropout(.20)(x);x=layers.Dense(64,activation='relu')(x)
    x=layers.Dropout(.10)(x);x=layers.Dense(1)(x)
    output=layers.Activation(lambda z:tf.nn.softplus(z)+1e-8,name='lambda_output')(x)
    net=keras.Model(inputs=inputs,outputs=output)
    net.compile(optimizer=keras.optimizers.Adam(learning_rate=.001,clipnorm=1),
                loss=keras.losses.Poisson(),metrics=[keras.metrics.MeanAbsoluteError(name='MAE')])
    train_inputs={'num_input':xn}
    test_inputs={'num_input':xt}
    for j,name in enumerate(CATS):
        train_inputs[name+'_input']=a[:,j:j+1]; test_inputs[name+'_input']=b[:,j:j+1]
    stopper=keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
    net.fit(train_inputs,ytrain.astype('float32'),validation_split=.2,epochs=50,batch_size=256,
            callbacks=[stopper],verbose=1)
    return np.clip(net.predict(test_inputs,batch_size=256,verbose=0).ravel(),1e-8,cap)

def main():
    pa=argparse.ArgumentParser();pa.add_argument('csv',nargs='?',default=str(DEFAULT))
    pa.add_argument('--mlp',action='store_true',help='Ejecuta también MLP de la V7 con TensorFlow instalado')
    pa.add_argument('--out',default=None)
    args=pa.parse_args(); root=Path(args.out) if args.out else Path(args.csv).parent/'validacion_adicional_V7'
    root.mkdir(exist_ok=True,parents=True)
    cats,nums,y,periods,groups=load_csv(args.csv)
    n=len(y)
    idx=np.arange(n)
    rng_train,rng_test=train_test_split(idx,test_size=.20,random_state=42,shuffle=True)
    splitter=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=42)
    g_train,g_test=next(splitter.split(idx,groups=groups))
    t_train=idx[np.isin(periods,['2025-09','2025-11'])]
    t_test=idx[periods=='2026-02']
    splits={'Aleatorio_80_20':(rng_train,rng_test),
            'Grupos_excluyentes_80_20':(g_train,g_test),
            'Temporal_2025_a_feb2026':(t_train,t_test)}
    results=[];stats=[]
    for name,(tr,te) in splits.items():
        tic=time.monotonic()
        enc=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
        xtr_c=enc.fit_transform(cats[tr]);xte_c=enc.transform(cats[te]);
        xtr=np.column_stack([xtr_c,nums[tr]]);xte=np.column_stack([xte_c,nums[te]])
        yy=y[tr]; yt=y[te]; cap=cap_lambda(yy)
        train_grp=set(groups[tr]); test_grp=set(groups[te]); overlaps=len(train_grp & test_grp)
        stats.append({'particion':name,'train':len(tr),'test':len(te),
            'positivos_train':int((yy>0).sum()),'positivos_test':int((yt>0).sum()),
            'prevalencia_train_pct':round(100*(yy>0).mean(),5),
            'prevalencia_test_pct':round(100*(yt>0).mean(),5),
            'media_train':float(yy.mean()),'media_test':float(yt.mean()),
            'grupos_compartidos':overlaps,'grupos_train':len(train_grp),'grupos_test':len(test_grp),
            'grupos_ausentes_en_train_en_test':len(test_grp-train_grp),
            'periodos_train':','.join(sorted(set(periods[tr]))),'periodos_test':','.join(sorted(set(periods[te]))),'cap_lambda':cap})
        all_models=[('Promedio_historico',DummyRegressor(strategy='mean')),
                    ('Random_Forest',RandomForestRegressor(n_estimators=80,max_depth=18,min_samples_leaf=5,random_state=42,n_jobs=4)),
                    ('HistGradientBoosting_Poisson',HistGradientBoostingRegressor(loss='poisson',learning_rate=.06,max_iter=120,max_leaf_nodes=31,l2_regularization=.1,random_state=42))]
        for model_name,m in all_models:
            start=time.monotonic()
            m.fit(xtr,yy); yp=np.clip(m.predict(xte),1e-8,cap)
            row=dict(particion=name,modelo=model_name,**metrics(yt,yp))
            row['minutos_entrenamiento_prediccion']=round((time.monotonic()-start)/60,3)
            row['cap_lambda']=cap
            results.append(row)
            print(name,model_name,'MAE',round(row['MAE'],5),'RMSE',round(row['RMSE'],5),'R2',round(row['R2'],5),'time',row['minutos_entrenamiento_prediccion'],flush=True)
        if args.mlp:
            start=time.monotonic()
            yp=mlp_pred(cats[tr],cats[te],nums[tr],nums[te],yy,cap)
            row=dict(particion=name,modelo='MLP_embedding_V7_reentrenada',**metrics(yt,yp))
            row['minutos_entrenamiento_prediccion']=round((time.monotonic()-start)/60,3)
            row['cap_lambda']=cap
            results.append(row)
            print(name,'MLP_embedding_V7_reentrenada','MAE',round(row['MAE'],5),'R2',round(row['R2'],5),flush=True)
        print('Fin',name,'segundos',round(time.monotonic()-tic,1),flush=True)
    for filename,items in [('01_metricas_particiones.csv',results),('02_caracteristicas_particiones.csv',stats)]:
        with (root/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    with (root/'03_metodologia.json').open('w',encoding='utf-8') as f:
        json.dump({'archivo':str(args.csv),'registros':n,'objetivo':Y_COL,
            'particiones':list(splits),'cap_lambda_entrenamiento':True,
            'notas':['Modelos reentrenados de manera independiente; resultados NO son métricas de la MLP original.',
                     'Validación temporal entrena en 2025-09 y 2025-11 y prueba en 2026-02.',
                     'Validación por grupos separa departamento-municipio-actividad entre entrenamiento y prueba.',
                     'Sin datos de futuros meses continuos ni garantías de estacionalidad.',
                     'OrdinalEncoder se ajusta SOLO en entrenamiento para evitar filtración categórica.']},f,ensure_ascii=False,indent=2)
    print('Archivos guardados en',root)
if __name__=='__main__':main()
