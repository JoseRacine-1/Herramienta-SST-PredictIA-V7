#!/usr/bin/env python3
"""Auditoría posentrenamiento de sensibilidad de la V7 usando solo predicciones exportadas.
No reentrena modelos ni modifica la base ni las predicciones originales.
"""
from pathlib import Path
from collections import defaultdict
import csv, json, time, re
import argparse
import numpy as np
from scipy.stats import spearmanr
from lxml import etree

parser=argparse.ArgumentParser(description='Sensibilidad del índice V7 a ponderaciones alternativas')
parser.add_argument('predicciones_xlsx',nargs='?',default='/mnt/data/rar_extracted/resultados_herramienta_sst_v7_dashboard/excel/10_predicciones_completas_herramienta_sst.xlsx')
parser.add_argument('--out',default=None)
args=parser.parse_args()
SRC=Path(args.predicciones_xlsx)
OUT=Path(args.out) if args.out else SRC.parent/'analisis_sensibilidad_V7'
OUT.mkdir(exist_ok=True,parents=True)

NS='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
from zipfile import ZipFile

def colindex(ref):
    s=re.match(r'[A-Z]+', ref).group(0)
    x=0
    for ch in s: x=x*26+ord(ch)-64
    return x-1

# 3 identificadores, 1 población, 6 predicciones, índice original, nivel y accidentes observados
req={'DPTO','MPIO','ACTIVEC','TRABAJADORES',
     'lambda_PRESUACCIDETRASUCE','lambda_MUERTES_REPOR_AT','lambda_NUEVAPENSIOINVA_R_AT',
     'lambda_NUEVAPENSIOINVA_R_EL','lambda_INCAPERMAPARCIAR_AT','lambda_INCAPERMAPARCIAR_EL',
     'indice_riesgo_compuesto','nivel_riesgo','PRESUACCIDETRASUCE'}
arrays=defaultdict(list)
t0=time.monotonic()
with ZipFile(SRC) as z, z.open('xl/worksheets/sheet1.xml') as f:
    header=None
    for _,row in etree.iterparse(f,events=('end',),tag=NS+'row'):
        vals={}
        for c in row:
            if not c.tag.endswith('}c'): continue
            j=colindex(c.get('r'))
            v=c.find(NS+'v')
            if v is not None: vv=v.text
            else:
                ins=c.find(NS+'is')
                vv=''.join(ins.itertext()) if ins is not None else ''
            if header is None or j in target_positions:
                vals[j]=vv
        if header is None:
            header={j:v for j,v in vals.items()}
            target_positions={j for j,n in header.items() if n in req}
            assert len(target_positions)==len(req),(target_positions,req)
        else:
            for j in target_positions:
                field=header[j]
                vv=vals.get(j)
                if field in ('DPTO','MPIO','ACTIVEC','nivel_riesgo'):
                    arrays[field].append(str(vv) if vv is not None else '')
                else:
                    arrays[field].append(float(vv) if vv not in (None,'') else np.nan)
        row.clear()
        while row.getprevious() is not None:
            del row.getparent()[0]
print('XLSX parsed rows',len(arrays['MPIO']),'seconds',round(time.monotonic()-t0,1))

keys=['DPTO','MPIO','ACTIVEC']
targs=['PRESUACCIDETRASUCE','MUERTES_REPOR_AT','NUEVAPENSIOINVA_R_AT',
      'NUEVAPENSIOINVA_R_EL','INCAPERMAPARCIAR_AT','INCAPERMAPARCIAR_EL']
P=np.column_stack([np.asarray(arrays['lambda_'+t],dtype=np.float64) for t in targs]); n=len(P)
T=np.array(arrays['TRABAJADORES'],dtype=float)
original=np.array(arrays['indice_riesgo_compuesto'],dtype=float)
q99=np.maximum(np.quantile(P,0.99,axis=0),1.0)
N=P/q99
w0=np.array([1,5,4,4,3,3],float)
base=N@w0
print('baseline diff',np.max(np.abs(base-original)),'scales',q99)
assert np.allclose(base,original,rtol=1e-8,atol=1e-5)

scenarios={
    'Oficial_V7':np.array([1,5,4,4,3,3],float),
    'Pesos_iguales':np.array([1,1,1,1,1,1],float),
    'Jerarquia_suavizada':np.array([1,4,3,3,2,2],float),
    'Muertes_mas_peso':np.array([1,8,4,4,3,3],float),
    'Solo_accidentes':np.array([1,0,0,0,0,0],float),
    'Sin_muertes_e_invalidez_AT':np.array([1,0,0,4,3,3],float),
}

# Agrupación exactamente como generate_rankings: DPTO+MPIO+ACTIVEC
# usando índices de grupo ordenados lexicográficamente
levels=np.array(arrays['nivel_riesgo'])
# np.unique over tuple arrays expensive; Python dictionary stable
ktuple=list(zip(*(arrays[k] for k in keys)))
idmap={}
group_ids=np.empty(n,dtype=np.int32)
rep_keys=[]
for i,k in enumerate(ktuple):
    if k not in idmap:
        idmap[k]=len(idmap);rep_keys.append(k)
    group_ids[i]=idmap[k]
g=len(idmap)
num=np.bincount(group_ids,minlength=g)
workers=np.bincount(group_ids,weights=T,minlength=g)
filter_tasa=(num>=2)&((workers/num)>=10)
acc_real=np.bincount(group_ids,weights=np.array(arrays['PRESUACCIDETRASUCE'],float),minlength=g)

# top de index con criterio estable de desempate = primer grupo visto

def top_indices(vals,k,eligible=None):
    idx=np.arange(len(vals),dtype=np.int64) if eligible is None else np.flatnonzero(eligible)
    # sort stable by score descending and identifier asc
    return idx[np.lexsort((idx,-vals[idx]))[:k]]

base_total=np.bincount(group_ids,weights=base,minlength=g)
base_rate=base_total/np.maximum(workers,1)*1000
base_top100=top_indices(base_total,100)
base_top100_rate=top_indices(base_rate,100,filter_tasa)
base_top20=top_indices(base_total,20)
base_top20_rate=top_indices(base_rate,20,filter_tasa)
base_classes=np.digitize(base,np.quantile(base,[0.6,0.9,0.97]),right=True)
assert np.mean(base_classes==np.array([{'Bajo':0,'Medio':1,'Alto':2,'Crítico':3}[v] for v in levels]))>0.99999

recap=[]
rank_rows=[]
level_rows=[]
for sc,w in scenarios.items():
    scores=N@w
    total=np.bincount(group_ids,weights=scores,minlength=g)
    rel=total/np.maximum(workers,1)*1000
    cut=np.quantile(scores,[0.60,0.90,0.97])
    cls=np.digitize(scores,cut,right=True)
    tops=top_indices(total,100);tops20=top_indices(total,20)
    tops_rel=top_indices(rel,100,filter_tasa);tops20_rel=top_indices(rel,20,filter_tasa)
    overlaps={
        'top100_total_compartidos':len(set(tops)&set(base_top100)),
        'top20_total_compartidos':len(set(tops20)&set(base_top20)),
        'top100_relativo_compartidos':len(set(tops_rel)&set(base_top100_rate)),
        'top20_relativo_compartidos':len(set(tops20_rel)&set(base_top20_rate)),
    }
    recap.append({'escenario':sc, 'registros':n, 'segmentos':g, 'segmentos_elegibles_relativo':int(filter_tasa.sum()),
         **overlaps,'cambio_nivel_pct':round(100*np.mean(cls!=base_classes),4),
         'cambio_alto_critico_pct':round(100*np.mean((cls>=2)!=(base_classes>=2)),4),
         'spearman_puntaje_registros':round(float(spearmanr(base,scores).statistic),6),
         'spearman_total_segmentos':round(float(spearmanr(base_total,total).statistic),6),
         'p60':cut[0],'p90':cut[1],'p97':cut[2]})
    for risklevel in range(4):
        level_rows.append({'escenario':sc,'nivel':['Bajo','Medio','Alto','Crítico'][risklevel],
                          'registros':int((cls==risklevel).sum()),'porcentaje':round(100*(cls==risklevel).mean(),3)})
    for typ,inds,arr in [('total',tops,total),('relativo_filtrado',tops_rel,rel)]:
        for r,ix in enumerate(inds,1):
            key=rep_keys[ix]
            rank_rows.append({'escenario':sc,'ranking':typ,'posicion':r,'DPTO':key[0],
                              'MPIO':key[1],'ACTIVEC':key[2],
                              'puntuacion':arr[ix],'registros':int(num[ix]),
                              'trabajadores_acumulados':int(workers[ix]),
                              'accidentes_observados':int(acc_real[ix])})
    print(sc,overlaps,'cambio nivel',round(100*np.mean(cls!=base_classes),2))

for name,rows in [('01_resumen_sensibilidad.csv',recap),('02_top100_escenarios.csv',rank_rows),('03_niveles_escenarios.csv',level_rows)]:
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

with (OUT/'04_metadatos_ejecucion.json').open('w',encoding='utf-8') as f:
    json.dump({'n':n,'grupos':g,'ponderaciones':{k:list(map(int,v)) for k,v in scenarios.items()},
               'p99_escalas':dict(zip(targs,q99.tolist())),
               'escenarios':recap,
               'aclaracion':'Análisis postentrenamiento sin modificar las predicciones; niveles con percentiles recalculados en cada escenario; filtros 2 observaciones y 10 trabajadores promedio; no valida etiquetas de severidad.'},f,ensure_ascii=False,indent=2)
print('DONE',round(time.monotonic()-t0,1),'seconds')
