# PSI on SPU
# Author: CAO YIMING

import secretflow as sf
import numpy as np
from sklearn.datasets import load_iris
import os
import pandas as pd


"""
First, we need a dataset for constructing vertical partitioned scenarios. 
For simplicity, we use iris dataset here. 
We add two columns to it for subsequent single-column and multi-column intersection demonstrations 
- uid：Sample unique ID. 
- month：Simulate a scenario where samples are generated monthly. The first 50% of the samples are generated in January, and the last 50% of the samples are generated in February.
""" 
data, target = load_iris(return_X_y=True, as_frame=True)
data['uid'] = np.arange(len(data)).astype('str')
data['month'] = ['Jan'] * 75 + ['Feb'] * 75

"""
We randomly sample the iris data three times to simulate the data provided by alice, bob, and carol, 
and the three data are in an unaligned state.
"""
os.makedirs('.data', exist_ok=True)
da, db, dc = data.sample(frac=0.9), data.sample(frac=0.8), data.sample(frac=0.7)

da.to_csv('.data/alice.csv', index=False)
db.to_csv('.data/bob.csv', index=False)
dc.to_csv('.data/carol.csv', index=False)


"""
Two parties PSI

We virtualize three logical devices on the physical device: 
- alice, bob: PYU device, responsible for the local plaintext computation of the participant. 
- spu：SPUdevice, consists of alice and bob, responsible for the ciphertext calculation of the two parties.
"""
sf.shutdown()

sf.init(['alice', 'bob', 'carol'], num_cpus=8, log_to_driver=False)
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

"""
Single-column PSI

Next, we use uid to intersect the two data, 
SPU provide psi_csv which take the csv file as input and generate the csv file after the intersection. 
The default protocol is KKRT.
"""
input_path = {alice: '.data/alice.csv', bob: '.data/bob.csv'}
output_path = {alice: '.data/alice_psi.csv', bob: '.data/bob_psi.csv'}
spu.psi_csv('uid', input_path, output_path)

# To check the correctness of the results, we use pandas.DataFrame.join to inner join da and db. 
# It can be seen that the two data have been aligned according to uid and sorted according to their lexicographical order.
df = da.join(db.set_index('uid'), on='uid', how='inner', rsuffix='_bob', sort=True)
expected = df[da.columns].astype({'uid': 'int64'}).reset_index(drop=True)

da_psi = pd.read_csv('.data/alice_psi.csv')
db_psi = pd.read_csv('.data/bob_psi.csv')

pd.testing.assert_frame_equal(da_psi, expected)
pd.testing.assert_frame_equal(db_psi, expected)

"""
Multi-columns PSI

We can also use multiple fields to intersect, the following demonstrates the use of uid and month to intersect two data. 
In terms of implementation, multiple fields are concatenated into a string, so please ensure that there is no duplication of the multi-column composite primary key.
"""
spu.psi_csv(['uid', 'month'], input_path, output_path)

# Similarly, we use pandas.DataFrame.join to verify the correctness of the result, 
# we can see that the two data have been aligned according to uid and month, and sorted according to their lexicographical order.
df = da.join(db.set_index(['uid', 'month']), on=['uid', 'month'], how='inner', rsuffix='_bob', sort=True)
expected = df[da.columns].astype({'uid': 'int64'}).reset_index(drop=True)

da_psi = pd.read_csv('.data/alice_psi.csv')
db_psi = pd.read_csv('.data/bob_psi.csv')

pd.testing.assert_frame_equal(da_psi, expected)
pd.testing.assert_frame_equal(db_psi, expected)


"""
Three parties PSI
"""
sf.shutdown()

sf.init(['alice', 'bob', 'carol'], num_cpus=8, log_to_driver=False)
alice, bob, carol = sf.PYU('alice'), sf.PYU('bob'), sf.PYU('carol')
spu_3pc = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob', 'carol']))

# Then, use uid and month as the composite primary key to perform a three-way negotiation. 
# It should be noted that the three-way negotiation only supports the ECDH protocol for the time being.
input_path = {alice: '.data/alice.csv', bob: '.data/bob.csv', carol: '.data/carol.csv'}
output_path = {alice: '.data/alice_psi.csv', bob: '.data/bob_psi.csv', carol: '.data/carol_psi.csv'}
spu_3pc.psi_csv(['uid', 'month'], input_path, output_path, protocol='ecdh')


# Similarly, we use pandas.DataFrame.join to verify the correctness of the result.
keys = ['uid', 'month']
df = da.join(db.set_index(keys), on=keys, how='inner', rsuffix='_bob', sort=True).join(
    dc.set_index(keys), on=keys, how='inner', rsuffix='_carol', sort=True)
expected = df[da.columns].astype({'uid': 'int64'}).reset_index(drop=True)

da_psi = pd.read_csv('.data/alice_psi.csv')
db_psi = pd.read_csv('.data/bob_psi.csv')
dc_psi = pd.read_csv('.data/carol_psi.csv')

pd.testing.assert_frame_equal(da_psi, expected)
pd.testing.assert_frame_equal(db_psi, expected)
pd.testing.assert_frame_equal(dc_psi, expected)
