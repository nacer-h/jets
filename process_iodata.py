import uproot
import pandas as pd
import fastjet as fj
import fjext
import numpy as np
import ROOT

## ---------------------------------------------------------
## Convert ROOT TTree to pandas DataFrame
## ---------------------------------------------------------

# import matplotlib.pyplot as plt
# import mplhelp
# mplhelp.histplot(data["PWGHF_TreeCreator/h_coutputEntriesHFTreeCreator_PbPb_5TeV"].to_boost())
# plt.hist(uproot.open("/rstorage/alice/data/LHC18qr/570/LHC18q/000296433/0068/AnalysisResults.root:PWGHF_TreeCreator/tree_Particle/ParticlePt"), bins=10, range=(0,1000))

data = uproot.open("/rstorage/alice/data/LHC18qr/570/LHC18q/000296433/0068/AnalysisResults.root")
print(data.classnames())
print(data["PWGHF_TreeCreator/tree_Particle"].typenames())
# print(data["PWGHF_TreeCreator/tree_Particle"].keys(filter_name="ev_id"))
# print(data["PWGHF_TreeCreator/tree_Particle"].keys(filter_typename="float"))
# print(np.array(data["PWGHF_TreeCreator/tree_Particle"])[:5])

# data["PWGHF_TreeCreator/coutputEntriesHFTreeCreator_PbPb_5TeV"].to_boost()

tree_Particle = uproot.open("/rstorage/alice/data/LHC18qr/570/LHC18q/000296433/0068/AnalysisResults.root:PWGHF_TreeCreator/tree_Particle")

print(tree_Particle)
tree_Particle.show()
# print(tree_Particle["ParticlePt"].array(library="pd"))
df_particles = tree_Particle.arrays(["run_number","ev_id","ev_id_ext","ev_id_long","ParticlePt","ParticleEta","ParticlePhi"], library="pd")
print(df_particles)

## group dataframe with the same "ev_id_ext"
df_particles_unique = df_particles.groupby([tree_Particle['run_number'],tree_Particle['ev_id'],tree_Particle['ev_id_ext']])
print(df_particles_unique.head())

## advantage of using awkward arrays over pandas is that awkward arrays handle better when you have different numbers of electrons and muons in the event. Where awkward can make array objects with different numbers for each of these things.

## jagged arrays: array with different length objects (ntuple that has different length branching), example: Pt of tracks in each vertex is different. technically speaking: usually the trees have same length branches (same entries), but could have varied lengths in the second dimension => jagged

## when working with data with diferrent number of elements in each event, e.g. number of jets and muons is different in each event. Pandas DataFrame has only one multi index, so need a two separate DataFrames for jets and muons, then join them at the end.


