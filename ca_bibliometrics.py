# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 09:57:44 2020

@author: Carlo
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import networkx as nx
import community
import json

import plotly.graph_objects as go
import plotly
import plotly.express as px

import re
from collections import Counter
from nltk.corpus import stopwords

import time

def graph_from_matrix(A, id_map, data, attrs=[]):
    """
    Funzione ripresa da networkx e modificata per permettere il recupero diretto di alcuni attributi dei nodi.
    """
    kind_to_python_type = {
        "f": float,
        "i": int,
        "u": int,
        "b": bool,
        "c": complex,
        "S": str,
        "V": "void",
    }
    kind_to_python_type["U"] = str
    
    G = nx.empty_graph(0)
    n, m = A.shape
    
    if n != m:
        raise nx.NetworkXError("Adjacency matrix is not square.", f"nx,ny={A.shape}")
    dt = A.dtype
    try:
        python_type = kind_to_python_type[dt.kind]
    except Exception:
        raise TypeError(f"Unknown numpy data type: {dt}")
        
    if 'hash' not in attrs:
        attrs.append('hash')
    
    ns = [(i, {'id_map':int(id_map[i]), **{x:list(data.loc[data['hash'] == id_map[i]][x])[0] for x in attrs}}) for i in range(n)]
    
    G.add_nodes_from(ns)
    
    edges = map(lambda e: (int(e[0]), int(e[1])), zip(*(np.asarray(A).nonzero())))
    triples = ((u, v, dict(weight=python_type(A[u, v]))) for u, v in edges)
    G.add_edges_from(triples)
    
    return G

def to_str_hash(obj):
    return hash(str(obj))

def load_data_and_hash(source):
    """
    Funzione per caricare il csv in un DataFrame, aggiunge la colonna 'hash' per avere un identificativo unico degli oggetti
    """
    dtype = {'ID': 'int64', 'PT': 'str', 'AU': 'str', 'AF': 'str', 'C1': 'str', 'EM': 'str', 'AA': 'str',
             'TI': 'str', 'PY': 'int64', 'SO': 'str', 'VL': 'str', 'IS': 'str', 'AR': 'str', 'BP': 'str',
             'EP': 'str', 'PG': 'int64', 'TC': 'int64', 'DI': 'str', 'LI': 'str', 'AB': 'str', 'DE': 'str',
             'DT': 'str', 'FS': 'str', 'UT': 'str', 'CR_ID': 'int64', 'CR': 'str', 'RPY': 'int', 'N_CR': 'int64',
             'PERC_YR': 'float64', 'PERC_ALL': 'float64', 'CR_AU': 'str', 'AU_L': 'str', 'AU_F': 'str',
             'AU_A': 'str', 'CR_TI': 'str', 'J': 'str', 'J_N': 'str', 'J_S': 'str', 'VOL': 'str', 'PAG': 'str',
             'DOI': 'str', 'CID2': 'str', 'CID_S': 'int64', 'N_PYEARS': 'int64', 'PERC_PYEARS': 'float64',
             'N_TOP50': 'int64', 'N_TOP25': 'int64', 'N_TOP10': 'int64', 'SEQUENCE': 'str', 'TYPE': 'str',
             'SEARCH_SCORE': 'int64', 'PM' : 'str', 'BP' : 'str', 'PD' : 'str', 'PU' : 'str', 'PN' : 'str',
             'SU' : 'str', 'PI' : 'str', 'SN' : 'str', 'OA' : 'str', 'CT' : 'str', 'GP' : 'str', 'VL' : 'str',
             'BN' : 'str', 'PT' : 'str', 'Z9' : 'str', 'BE' : 'str', 'AF' : 'str', 'CL' : 'str', 'IS' : 'str',
             'U2' : 'str', 'J9' : 'str', 'SE' : 'str', 'WC' : 'str', 'EI' : 'str', 'CY' : 'str', 'PA' : 'str',
             'SP' : 'str', 'EP' : 'str', 'DA' : 'str', 'LA' : 'str', 'FX' : 'str', 'DE' : 'str', 'U1' : 'str',
             'FU' : 'str', 'NR' : 'str', 'GA' : 'str', 'OI' : 'str', 'UT' : 'str', 'EM' : 'str', 'HO' : 'str',
             'KP' : 'str', 'RP' : 'str', 'SI' : 'str', 'SC' : 'str', 'RI' : 'str', 'JI' : 'str' }
    
    data = pd.read_csv(source, header=0, dtype=dtype, index_col = ["ID","CR_ID"])
    
    data['hash'] = data.apply(to_str_hash, axis=1)
    
    ls = len(list(data['hash']))
    st = len(list(set(data['hash'])))
    
    if ls != st:
        print("Warning: hash collision, {} elements vs {} hashes".format(ls,st))
    
    return data

def gen_adj_matrix(data, atype):
    """
    Genero la matrice di adiacenza a partire dal DataFrame,
    di default per il bibliographic coupling, se transpose è vero per co-citation
    """
    links = np.array(list(data.index.values))
    adj = np.zeros((links[:,0].max(),links[:,1].max()))
    links-=1
    #citation matrix (rows: citing, cols: cited reference)
    adj[(links[:,0],links[:,1])] = 1
    #adjacency matrix
    if atype == 'co':
        mat = np.dot(adj.T,adj)
    else:
        mat = np.dot(adj,adj.T)
    np.fill_diagonal(mat, 0)
    return mat

def gen_adj_matrix_and_map(data, atype):
    """
    Genero la matrice di adiacenza a partire dal DataFrame,
    di default per il bibliographic coupling, se transpose è vero per co-citation
    """
    links = np.array(list(data.index.values))
    cit = len(list(set(links[:,0])))
    ref = len(list(set(links[:,1])))
    adj = np.zeros((cit,ref))
    
    cits_v = np.arange(cit)
    refs_v = np.arange(ref)
    cits_k = list(dict.fromkeys(links[:,0]))
    refs_k = list(dict.fromkeys(links[:,1]))
    
    cits = dict(zip(cits_k, cits_v))
    refs = dict(zip(refs_k, refs_v))
    
    rows = [cits[i] for i in links[:,0]]
    cols = [refs[i] for i in links[:,1]]
    
    #citation matrix (rows: citing, cols: cited reference)
    adj[(rows,cols)] = 1
    #adjacency matrix
    if atype == 'co':
        mat = np.dot(adj.T,adj)
        id_map = [list(data.reorder_levels([1,0]).loc[i]['hash'])[0] for i in links[:,1]]
    else:
        #bc
        mat = np.dot(adj,adj.T)
        id_map = [list(data.loc[i]['hash'])[0] for i in links[:,0]]
    id_map = list(dict.fromkeys(id_map))
    np.fill_diagonal(mat, 0)
    
    return (mat, id_map)

def reduce(matrix, min_degree, id_map):
    """
    Rimuovo dalla matrice quegli elementi che hanno grado minore di min_degree
    NB: restano valori minori di min_degree, ma solo quelli che prima erano maggiori
    e diventano più piccoli a causa della rimozione di altre righe-colonne!
    """
    
    #id_map = np.arange(0,matrix.shape[0])+1
    id_map = np.array(id_map)
    rem = np.where(matrix.sum(axis=0) < min_degree)
    
    matrix = np.delete(matrix, rem, axis=0)
    matrix = np.delete(matrix, rem, axis=1)
    id_map = np.delete(id_map, rem)
    
    return (matrix, id_map)

def list_to_nodes(G, ldiz, laname):
    """
    Prende un grafo, una lista di dizionari di attributo ai nodi,
    e una lista di stringhe di nomi di attributi:
    restituisce grafo aggiornato con gli attributi assegnati ai nodi.
    """
    for i in G.nodes:
        for x,d in enumerate(ldiz):
            G.nodes[i][laname[x]] = ldiz[x][i]
    return G

def degree_distribution(mat, show=False):
    s = mat.sum(axis=0)
    distr = plt.hist(s, bins=np.arange(min(s),max(s)+1))
    if show:
        plt.show()
    plt.close()
    return distr

def vis_slice(G):
    
    ws = list(set([i[2]['weight'] for i in G.edges(data=True)]))
    
    edge_traces = []
    
    for w in ws:
        edge_x = []
        edge_y = []
        weight = []
        for edge in G.edges(data=True):
            if edge[2]['weight'] == w:
                x0, y0 = G.nodes[edge[0]]['pos']
                x1, y1 = G.nodes[edge[1]]['pos']
                weight.append(edge[2]['weight'])
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)
        
        edge_traces.append(go.Scattergl(
            x=edge_x, y=edge_y,
            line=dict(width=w, color='#888'),
            hoverinfo='none',
            mode='lines'))
    
    node_x = []
    node_y = []
    node_color = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        node_color.append(G.nodes[node]['community'])
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=node_color,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    return (edge_traces, node_trace)

def edge_pos_zip(frm, to):
    l = zip(list(frm), list(to))
    return [i for sub in l for i in sub]

def caption_row(df, tm, atype):
    if atype == 'bc':
        return '{}<br>Degree: {}<br>Betweenness centrality: {}'.format(df['rec_id'],df['deg_{}'.format(tm)],df['centr_{}'.format(tm)])
    else:
        return '{}<br>Degree: {}<br>Betweenness centrality: {}'.format(df['rec_id'],df['deg_{}'.format(tm)],df['centr_{}'.format(tm)])

def caption(df, tm, atype):
    caption = df.apply(caption_row, args=[tm, atype], axis=1)
    return caption

def network_analyse(atype, betw_approx, min_degree, opts, time_span, time_step, weighted, out_file, tit):
    
    infos = dict()
    
    time_min = time_span[0]
    time_max = np.flip(np.arange(time_min, time_span[1]+1, time_step))
    
    max_deg = 0
    max_part = 0
    
    for ti, tm in enumerate(time_max):
        
        infos[(time_min,tm)] = dict()
    
        df = tot_df.query('PY >= {} and PY <= {}'.format(time_min, tm))
        # id_map va generato già alla creazione della matrice e poi eliminati i valori inutili in reduce
        
        # genero matrice di adiacenza
        adj, id_map = gen_adj_matrix_and_map(df, atype)
        
        if not weighted:
            adj[adj>0] = 1
        
        # riduco matrice eliminando elementi con grado < tot
        # ottengo matrice ridotta e mappa degli id
        adj, id_map = reduce(adj, min_degree, id_map)
        
        #deg_distr = degree_distribution(adj, False)
        
        # definisco gli attributi che voglio nei nodi del grafo
        if atype == 'bc':
            attrs = ["TI","SO","AU","KP","SC","WC","PY","rec_id"]
        else:
            attrs = ["TI","SO","AU","KP","SC","WC","PY","rec_id","CR"]
        # creo grafo
        G = graph_from_matrix(adj, id_map, df, attrs)
        
        if ti == 0:
            df_viz = pd.DataFrame({'hash': [G.nodes[i]['hash'] for i in G.nodes]})
        
        # informazioni generali sul grafo
        infos[(time_min,tm)]['nodes'] = G.order()
        infos[(time_min,tm)]['edges'] = G.size()
        infos[(time_min,tm)]['average_degree'] = G.size()/G.order()
        infos[(time_min,tm)]['average_weighted_degree'] = G.size('weight')/G.order()
        
        # layout per visualizzazione
        #pos = nx.spring_layout(G, **opts)
        if ti==0:
            _pos = nx.kamada_kawai_layout(G, **opts)
            posx = {G.nodes[i]['hash'] : _pos[i][0] for i in _pos}
            posy = {G.nodes[i]['hash'] : _pos[i][1] for i in _pos}
            py = {G.nodes[i]['hash'] : G.nodes[i]['PY'] for i in G.nodes}
            df_viz['x'] = df_viz['hash'].map(posx)
            df_viz['y'] = df_viz['hash'].map(posy)
            df_viz['PY'] = df_viz['hash'].map(py)
            df_viz['rec_id'] = df_viz['hash'].map({G.nodes[i]['hash'] : re.sub(r', DOI .*', '', G.nodes[i]['rec_id'] if atype=='bc' else G.nodes[i]['CR']) for i in G.nodes})
            df_edges = pd.DataFrame([{'from_h' : G.nodes[edge[0]]['hash'],
                                      'to_h' : G.nodes[edge[1]]['hash'],
                                      'from_x' : posx[G.nodes[edge[0]]['hash']],
                                      'from_y' : posy[G.nodes[edge[0]]['hash']],
                                      'to_x' : posx[G.nodes[edge[1]]['hash']],
                                      'to_y' : posy[G.nodes[edge[1]]['hash']],
                                      'weight' : edge[2]['weight']} for edge in G.edges(data=True)])
        
        # clustering
        part = community.best_partition(G)
        
        v_part = {G.nodes[i]['hash'] : part[i] for i in part}
        df_viz['part_{}'.format(tm)] = df_viz['hash'].map(v_part)
        
        # calcolo modularità della partizione
        comm = [{x for x in part if part[x] == i} for i in list(set(part.values()))]
        infos[(time_min,tm)]['modularity'] = nx.algorithms.community.quality.modularity(G, comm)
        infos[(time_min,tm)]['partition'] = [[G.nodes[x]['hash'] for x in part if part[x] == i] for i in list(set(part.values()))]
        
        # calcolo centralità dei nodi
        # nds è la soglia di approssimazione (per non appesantire troppo il calcolo)
        nds = min(betw_approx,len(G.nodes())) if betw_approx else None
        centr = nx.betweenness_centrality(G, weight='weight', normalized=True, k=nds)
        
        v_centr = {G.nodes[i]['hash'] : centr[i] for i in centr}
        df_viz['centr_{}'.format(tm)] = df_viz['hash'].map(v_centr)
        
        deg = dict(G.degree(weight='weight'))
        v_deg = {G.nodes[i]['hash'] : deg[i] for i in G.nodes}
        df_viz['deg_{}'.format(tm)] = df_viz['hash'].map(v_deg)
        
        max_deg = max(max(deg.values()), max_deg)
        max_part = max(len(comm), max_part)
        
        print('{} done'.format(tm))
        
        # assegno attributi ai nodi
        #G = list_to_nodes(G, [centr,part], ['centrality','community'])
            
    # ----------------------- #
    # --- VISUALIZZAZIONE --- #
    # ----------------------- #
    
    print("Exporting figure")
    
    # Create figure
    fig = go.Figure()
    
    # Add traces, one for each slider step
    time_max = np.flip(time_max)
    
    cmap = plt.get_cmap('jet')
    colors = [[i/(max_part-1), 'rgb({},{},{})'.format(*list(np.array(cmap(i/(max_part-1))[0:3])*255))] for i in range(max_part)]
    
    for ti, tm in enumerate(time_max):
        
        df = df_viz.query('PY >= {} and PY <= {}'.format(time_min, tm))
        df = df[df['part_{}'.format(tm)].notna()]
        
        dfe = df_edges.loc[df_edges['from_h'].isin(list(df['hash'])) & df_edges['to_h'].isin(list(df['hash']))]
        
        fig.add_trace(
            go.Scatter(
                    name="Edges",
                    x=edge_pos_zip(dfe['from_x'], dfe['to_x']), y=edge_pos_zip(dfe['from_y'], dfe['to_y']),
                    line=dict(width=0.05, color='#888'),
                              hoverinfo='none',
                              mode='lines')
                    )
        
        fig.add_trace(
                go.Scatter(
                    name="Nodes",
                    x=df['x'], y=df['y'],
                    mode='markers',
                    hoverinfo='text',
                    text=caption(df,tm,atype),
                    marker=dict(
                        color=df['part_{}'.format(tm)],
                        colorscale=colors,
                        size=5+(np.array(df['deg_{}'.format(tm)])/max_deg)*30,
                        line_width=1))
                    )
    
    # Create and add slider
    steps = []
    for i in np.arange(0, len(fig.data), 2):
        step = dict(
            label=str(time_max[i//2]),
            method="restyle",
            args=["visible", [False] * len(fig.data)],
        )
        step["args"][1][i] = True  # Toggle i'th trace to "visible"
        step["args"][1][i+1] = True  # Toggle i+1'th trace to "visible"
        steps.append(step)
    
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Year: "},
        pad={"t": 50},
        steps=steps
    )]
    
    fig.update_layout(
        sliders=sliders,
        title=tit,
        legend={'itemsizing':'constant'}
    )
    
    for i in range(len(fig.data)):
        fig.data[i].visible = False    
    
    fig.data[0].visible = True
    fig.data[1].visible = True
    
    fig.update_xaxes(range=[-700, 700], showticklabels=False)
    fig.update_yaxes(range=[-700, 700], showticklabels=False)
    
    plotly.offline.plot(fig, filename=out_file+'.html')
    
    return infos

def sankey_diagram(df, fname):
    nodes = list(set(list(df['source']) + list(df['target'])))
    end_yrs = np.array([eval(i)[1] for i in nodes])
    cns = np.array([eval(i)[2] for i in nodes])
    
    orig = [(0 if i in list(df['target']) else 1) for i in nodes]
    orig_n = sum(orig)
    
    cmap = plt.get_cmap('jet')
    
    srcs = []
    trgt = []
    vals = []
    
    origs = 0
    for i,x in enumerate(nodes):
        
        froms = df.loc[df['source'] == x]
        
        for y in froms.to_dict('records'):
            srcs.append(nodes.index(x))
            trgt.append(nodes.index(y['target']))
            vals.append(int(y['value']))
            
        if orig[i] == 1:
            orig[i] = cmap(origs/orig_n)
            origs +=1        
    
    def color(i):
        if orig[i] == 0:
            tos = df.loc[df['target'] == nodes[i]]
            colors = []
            for x in list(tos['source']):
                y = nodes.index(x)
                if orig[y] == 0:
                    return 0
                else:
                    colors.append((orig[y], list(tos.loc[tos['source'] == x]['value'])[0]))
            res = [0,0,0,0]
            for c in colors:
                res[0] += (c[0][0] * c[1])
                res[1] += (c[0][1] * c[1])
                res[2] += (c[0][2] * c[1])
                res[3] += c[1]
            orig[i] = (res[0]/res[3], res[1]/res[3], res[2]/res[3], 0)
    
    for i in range(20):
        for x in range(len(nodes)):
            color(x)
    
    colors = ['rgb({},{},{})'.format(i[0]*255, i[1]*255, i[2]*255) for i in orig]
    
    fig = go.Figure(go.Sankey(
        arrangement = "snap",
        node = {
            "label": cns,
            "x": ((end_yrs-min(end_yrs))/max(end_yrs-min(end_yrs))),
            "y": ((cns/max(cns)) * 0.8) + 0.1,
            'pad':10,
            'color' : colors,
            'customdata' : ['Cluster: <b>{}</b>, year: <b>{}</b>'.format(cns[i],end_yrs[i]) for i in range(len((nodes)))],
            'hovertemplate' : '%{customdata}<extra></extra>'},
        link = {
            "source": srcs,
            "target": trgt,
            "value":  vals,
            'hovertemplate' : 'From cluster <b>%{source.label}</b> to <b>%{target.label}</b><br>Value <b>%{value}</b><extra></extra>'}))
    
    fig.update_layout(
        autosize=False,
        width=1024,
        height=768,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=10
        ),
    )
    
    plotly.offline.plot(fig, filename=fname+'.html')
    return fig

def create_sankey(data, fname):
    time_span = data['params']['time_span']
    time_step = data['params']['time_step']
    time_max = np.arange(time_span[0], time_span[1]+1, time_step)
    
    sank = dict()
    
    for t in time_max:
        if t < time_max[-1]:
            for cl_n, cl in enumerate(data['infos'][(time_span[0],t)]['partition']):
                for cl_n1, cl1 in enumerate(data['infos'][(time_span[0],t+time_step)]['partition']):
                    if type(sank.get((time_span[0],t,cl_n),0)) == type(0):
                        sank[(time_span[0],t,cl_n)] = dict()
                    sank[(time_span[0],t,cl_n)][(time_span[0],t+time_step,cl_n1)] = sank[(time_span[0],t,cl_n)].get((time_span[0],t+time_step,cl_n1),0) + len([i for i in cl if i in cl1])
    
    sank_list = []
    for i in sank:
        for x in sank[i]:
            if sank[i][x] > 0:
                sank_list.append({'source': str(i), 'target': str(x), 'value' : sank[i][x]})
    
    df = pd.DataFrame(sank_list)
    fig = sankey_diagram(df, fname)
    return fig


def cluster_analyse(data, attrs, tot_df):
    res = dict()
    for i, part in enumerate(data):
        df = tot_df.loc[tot_df['hash'].isin(part)]
        res[i] = dict()
        for attr in attrs:
            if attr in ['TI','AB']:
                text = ' '.join(list(df[attr].dropna().astype(str)))
                words = re.findall(r'\w+', text.lower())
                ignore = list(stopwords.words('english'))
                res[i][attr] = Counter(w for w in words if w not in ignore)
            elif attr in ['AU']:
                text = '; '.join(list(df[attr].dropna().astype(str)))
                words = text.lower().split('; ')
                res[i][attr] = Counter(words)
            elif attr in ['CR_AU','AU_L','J_N','J_S', 'SO']:
                text = list(df[attr].dropna().astype(str))
                words = [j.lower().replace('.','') for j in text]
                #words = [(j.lower().replace('.','').split(' '))[0] for j in text]
                res[i][attr] = Counter(words)
            elif attr in ['KP','SC','WC','DE']:
                text = list(df[attr].dropna().astype(str))
                text = [eval(j.lower()) for j in text]
                words = [item for sublist in text for item in sublist]
                res[i][attr] = Counter(words)
            else:
                #print('"{}" generic attribute'.format(attr))
                res[i][attr] = Counter(list(df[attr].dropna()))
    return res

def treemap(data, attr, threshold, fname, reverse=False):
    
    if reverse:
        threshold = 0
    
    df_source = []
    for i in data:
        other = 0
        for x in data[i][attr]:
            if data[i][attr][x] > threshold:
                df_source.append({'cluster': i, attr: x, 'value': data[i][attr][x]})
            else:
                other += data[i][attr][x]
        if other > 0:
            df_source.append({'cluster': i, attr: 'other', 'value': other})
    df = pd.DataFrame(df_source)
    
    df["all"] = "all"
    df["cluster_percent"] = df.apply(lambda x: np.around(100 * (x["value"]/df.loc[(df['cluster'] == x['cluster'])]['value'].sum()), 2), axis=1)
    
    if reverse:
        fig = px.treemap(df,
                         path=['all', attr, 'cluster'],
                         values='value',
                         color=attr,
                         hover_data=['cluster_percent'],
                         color_discrete_sequence=px.colors.qualitative.Light24,
                         color_discrete_map={'(?)':'grey'})
    else:
        fig = px.treemap(df,
                         path=['all', 'cluster', attr],
                         values='value',
                         color=attr,
                         hover_data=['cluster_percent'],
                         color_discrete_sequence=px.colors.qualitative.Light24,
                         color_discrete_map={'(?)':'grey'})
    plotly.offline.plot(fig, filename=fname+'_'+attr+'_treemap.html')

# -------------- #
# --- SCRIPT --- #
# -------------- #
    
# carico i dati
path = 'D:/Mega/tesi/python/'

tot_df = load_data_and_hash(path+'data_1985_1995_recovered.csv')
print("Data loaded")

infos = []

params = dict(
    atype = 'bc',
    betw_approx = 10,
    min_degree = 10,
    opts = {'scale': 1000},
    time_span = (1985,1995),
    time_step = 1,
    weighted = True,
    out_file='bc_weighted_network',
    tit='Bibliographic coupling weighted network of cellular automata papers')

inf = network_analyse(**params)
infos.append({'params':params, 'infos':inf})

clusters_comp = cluster_analyse(infos[0]['infos'][params['time_span']]['partition'], ['TI','AU','AB','SO','CR','CR_AU','KP','WC','SC','DE'], tot_df)
treemap(clusters_comp, 'WC', 2, params['out_file']+'_'+str(params['time_span'][1]))
treemap(clusters_comp, 'SC', 2, params['out_file']+'_'+str(params['time_span'][1]))
treemap(clusters_comp, 'SO', 2, params['out_file']+'_'+str(params['time_span'][1]))
treemap(clusters_comp, 'TI', 2, params['out_file']+'_'+str(params['time_span'][1]))
treemap(clusters_comp, 'SC', 0, 'reversed_'+params['out_file']+'_'+str(params['time_span'][1]), True)

for i in infos:
    create_sankey(i, 'sankey_'+i['params']['out_file'])