"""
Home Credit Python Scoring Library and Workflow
Copyright © 2017-2019, Pavel Sůva, Marek Teller, Martin Kotek, Jan Zeller, 
Marek Mukenšnabl, Kirill Odintsov, Elena Kuchina, Jan Coufalík, Jan Hynek and
Home Credit & Finance Bank Limited Liability Company, Moscow, Russia.
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# -*- coding: utf-8 -*-

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

def transfer(X_source, X_target, nbr=50, normalize=True):
    if normalize:
        s=StandardScaler()
        s.fit(X_source)
        X_source = s.transform(X_source)
        X_target = s.transform(X_target)
    nbrs = NearestNeighbors(n_neighbors=nbr, algorithm='ball_tree')
    nbrs.fit(X_source)
    distances, neighbors=nbrs.kneighbors(X_source, nbr, True)
    print('fit1')
    nbrs2 = NearestNeighbors(algorithm='ball_tree')
    nbrs2.fit(X_target)
    
    weights=np.ndarray(shape=(X_source.shape[0],))

    for i in range(X_source.shape[0]):
        if i%1000==0:
            print(i)
        neighbors2=nbrs2.radius_neighbors([X_source[i, :]], [distances[i][nbr-1]], False)
        weights[i]=len(neighbors2[0])
    return weights
    