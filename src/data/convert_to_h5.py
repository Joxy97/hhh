import logging
from pathlib import Path

import awkward as ak
import click
import h5py
import numpy as np
import uproot
import vector
from coffea.nanoevents import BaseSchema, NanoEventsFactory

vector.register_awkward()

logging.basicConfig(level=logging.INFO)

N_JETS = 10
N_FJETS = 3
N_LEP = 2
N_TAU = 2

N_MASSES = 10
MIN_JET_PT = 20
MIN_FJET_PT = 200
MIN_JETS = 4
MIN_MASS = 50
PROJECT_DIR = Path(__file__).resolve().parents[3]


mappings = {'GluGluToHHHTo6B' : 0, 
            'QCD' : 1,
            'TT'  : 2,
            'WJets': 3,
            'ZJets': 3,
            'ZZ' : 4,
            'WW' : 4,
            'WZ' : 4,
            'GluGluToHHHTo4B' : 5,
            'GluGluToHHTo4B': 6,
            'GluGluToHHTo2B' : 7,
            'DYJetsToLL' : 8,
}


def get_n_features(name, events, iterator):
    if name.format(i=iterator[0]) not in dir(events):
        logging.warning(f"Variable {name.format(i=iterator[0])} does not exist in tree; returning all 0s")
        return ak.from_numpy(np.zeros((len(events), len(iterator))))
    return ak.concatenate(
        [np.expand_dims(events[name.format(i=i)], axis=-1) for i in iterator],
        axis=-1,
    )


def get_datasets(events):

    # small-radius jet info
    pt = get_n_features("jet{i}Pt", events, range(1, N_JETS + 1))
    bregcorr = get_n_features("jet{i}bRegCorr", events, range(1, N_JETS + 1))
    ptcorr = ptcorr = pt * bregcorr 
    eta = get_n_features("jet{i}Eta", events, range(1, N_JETS + 1))
    phi = get_n_features("jet{i}Phi", events, range(1, N_JETS + 1))
    #btag = get_n_features("jet{i}DeepFlavB", events, range(1, N_JETS + 1))
    btag = get_n_features("jet{i}PNetB", events, range(1, N_JETS + 1))
    jet_id = get_n_features("jet{i}JetId", events, range(1, N_JETS + 1))
    higgs_idx = get_n_features("jet{i}HiggsMatchedIndex", events, range(1, N_JETS + 1))
    hadron_flavor = get_n_features("jet{i}HadronFlavour", events, range(1, N_JETS + 1))
    matched_fj_idx = get_n_features("jet{i}FatJetMatchedIndex", events, range(1, N_JETS + 1))
    inv_mass = get_n_features("jet{i}Mass", events, range(1, N_JETS + 1))

    # paired masses
    mass1 = get_n_features("massjet1jet{i}", events, range(2, N_JETS + 1))
    mass2 = get_n_features("massjet2jet{i}", events, range(3, N_JETS + 1))
    mass3 = get_n_features("massjet3jet{i}", events, range(4, N_JETS + 1))
    mass4 = get_n_features("massjet4jet{i}", events, range(5, N_JETS + 1))
    mass5 = get_n_features("massjet5jet{i}", events, range(6, N_JETS + 1))
    mass6 = get_n_features("massjet6jet{i}", events, range(7, N_JETS + 1))
    mass7 = get_n_features("massjet7jet{i}", events, range(8, N_JETS + 1))
    mass8 = get_n_features("massjet8jet{i}", events, range(9, N_JETS + 1))
    mass9 = get_n_features("massjet9jet{i}", events, range(10, N_JETS + 1))

    pt1 = get_n_features("ptjet1jet{i}", events, range(2, N_JETS + 1))
    pt2 = get_n_features("ptjet2jet{i}", events, range(3, N_JETS + 1))
    pt3 = get_n_features("ptjet3jet{i}", events, range(4, N_JETS + 1))
    pt4 = get_n_features("ptjet4jet{i}", events, range(5, N_JETS + 1))
    pt5 = get_n_features("ptjet5jet{i}", events, range(6, N_JETS + 1))
    pt6 = get_n_features("ptjet6jet{i}", events, range(7, N_JETS + 1))
    pt7 = get_n_features("ptjet7jet{i}", events, range(8, N_JETS + 1))
    pt8 = get_n_features("ptjet8jet{i}", events, range(9, N_JETS + 1))
    pt9 = get_n_features("ptjet9jet{i}", events, range(10, N_JETS + 1))

    eta1 = get_n_features("etajet1jet{i}", events, range(2, N_JETS + 1))
    eta2 = get_n_features("etajet2jet{i}", events, range(3, N_JETS + 1))
    eta3 = get_n_features("etajet3jet{i}", events, range(4, N_JETS + 1))
    eta4 = get_n_features("etajet4jet{i}", events, range(5, N_JETS + 1))
    eta5 = get_n_features("etajet5jet{i}", events, range(6, N_JETS + 1))
    eta6 = get_n_features("etajet6jet{i}", events, range(7, N_JETS + 1))
    eta7 = get_n_features("etajet7jet{i}", events, range(8, N_JETS + 1))
    eta8 = get_n_features("etajet8jet{i}", events, range(9, N_JETS + 1))
    eta9 = get_n_features("etajet9jet{i}", events, range(10, N_JETS + 1))

    phi1 = get_n_features("phijet1jet{i}", events, range(2, N_JETS + 1))
    phi2 = get_n_features("phijet2jet{i}", events, range(3, N_JETS + 1))
    phi3 = get_n_features("phijet3jet{i}", events, range(4, N_JETS + 1))
    phi4 = get_n_features("phijet4jet{i}", events, range(5, N_JETS + 1))
    phi5 = get_n_features("phijet5jet{i}", events, range(6, N_JETS + 1))
    phi6 = get_n_features("phijet6jet{i}", events, range(7, N_JETS + 1))
    phi7 = get_n_features("phijet7jet{i}", events, range(8, N_JETS + 1))
    phi8 = get_n_features("phijet8jet{i}", events, range(9, N_JETS + 1))
    phi9 = get_n_features("phijet9jet{i}", events, range(10, N_JETS + 1))

    dr1 = get_n_features("drjet1jet{i}", events, range(2, N_JETS + 1))
    dr2 = get_n_features("drjet2jet{i}", events, range(3, N_JETS + 1))
    dr3 = get_n_features("drjet3jet{i}", events, range(4, N_JETS + 1))
    dr4 = get_n_features("drjet4jet{i}", events, range(5, N_JETS + 1))
    dr5 = get_n_features("drjet5jet{i}", events, range(6, N_JETS + 1))
    dr6 = get_n_features("drjet6jet{i}", events, range(7, N_JETS + 1))
    dr7 = get_n_features("drjet7jet{i}", events, range(8, N_JETS + 1))
    dr8 = get_n_features("drjet8jet{i}", events, range(9, N_JETS + 1))
    dr9 = get_n_features("drjet9jet{i}", events, range(10, N_JETS + 1))


    # large-radius jet info
    fj_pt = get_n_features("fatJet{i}Pt", events, range(1, N_FJETS + 1))
    fj_eta = get_n_features("fatJet{i}Eta", events, range(1, N_FJETS + 1))
    fj_phi = get_n_features("fatJet{i}Phi", events, range(1, N_FJETS + 1))
    fj_mass = get_n_features("fatJet{i}Mass", events, range(1, N_FJETS + 1))
    fj_sdmass = get_n_features("fatJet{i}MassSD", events, range(1, N_FJETS + 1))
    fj_regmass = get_n_features("fatJet{i}MassRegressed", events, range(1, N_FJETS + 1))
    fj_nsub = get_n_features("fatJet{i}NSubJets", events, range(1, N_FJETS + 1))
    fj_tau32 = get_n_features("fatJet{i}Tau3OverTau2", events, range(1, N_FJETS + 1))
    fj_xbb = get_n_features("fatJet{i}PNetXbb", events, range(1, N_FJETS + 1))
    fj_xqq = get_n_features("fatJet{i}PNetXjj", events, range(1, N_FJETS + 1))
    fj_qcd = get_n_features("fatJet{i}PNetQCD", events, range(1, N_FJETS + 1))
    fj_higgs_idx = get_n_features("fatJet{i}HiggsMatchedIndex", events, range(1, N_FJETS + 1))

    # leptons
    lep_pt = get_n_features("lep{i}Pt", events, range(1, N_LEP + 1))
    lep_eta = get_n_features("lep{i}Eta", events, range(1, N_LEP + 1))
    lep_phi = get_n_features("lep{i}Phi", events, range(1, N_LEP + 1))
    # taus
    tau_pt = get_n_features("tau{i}Pt", events, range(1, N_TAU + 1))
    tau_eta = get_n_features("tau{i}Eta", events, range(1, N_TAU + 1))
    tau_phi = get_n_features("tau{i}Phi", events, range(1, N_TAU + 1))



    #if events.signal == 1:
    #    signal = ak.from_numpy(np.ones(len(events), dtype = int))
    #else:
    #    signal = ak.from_numpy(np.zeros(len(events), dtype = int))

    signal = ak.from_numpy(np.full(len(events),events.signal,dtype = int))

    # keep events with >= MIN_JETS small-radius jets
    mask = ak.num(pt[pt > MIN_JET_PT]) >= MIN_JETS

    nprobejets = events.nprobejets
    mask = nprobejets >= 1 

    pt = pt[mask]
    ptcorr = ptcorr[mask]
    eta = eta[mask]
    phi = phi[mask]
    btag = btag[mask]
    jet_id = jet_id[mask]
    higgs_idx = higgs_idx[mask]
    hadron_flavor = hadron_flavor[mask]
    matched_fj_idx = matched_fj_idx[mask]
    inv_mass = inv_mass[mask]

    mass1 = mass1[mask]
    mass2 = mass2[mask]
    mass3 = mass3[mask]
    mass4 = mass4[mask]
    mass5 = mass5[mask]
    mass6 = mass6[mask]
    mass7 = mass7[mask]
    mass8 = mass8[mask]
    mass9 = mass9[mask]

    pt1 = pt1[mask]
    pt2 = pt2[mask]
    pt3 = pt3[mask]
    pt4 = pt4[mask]
    pt5 = pt5[mask]
    pt6 = pt6[mask]
    pt7 = pt7[mask]
    pt8 = pt8[mask]
    pt9 = pt9[mask]

    eta1 = eta1[mask]
    eta2 = eta2[mask]
    eta3 = eta3[mask]
    eta4 = eta4[mask]
    eta5 = eta5[mask]
    eta6 = eta6[mask]
    eta7 = eta7[mask]
    eta8 = eta8[mask]
    eta9 = eta9[mask]

    phi1 = phi1[mask]
    phi2 = phi2[mask]
    phi3 = phi3[mask]
    phi4 = phi4[mask]
    phi5 = phi5[mask]
    phi6 = phi6[mask]
    phi7 = phi7[mask]
    phi8 = phi8[mask]
    phi9 = phi9[mask]

    dr1 = dr1[mask]
    dr2 = dr2[mask]
    dr3 = dr3[mask]
    dr4 = dr4[mask]
    dr5 = dr5[mask]
    dr6 = dr6[mask]
    dr7 = dr7[mask]
    dr8 = dr8[mask]
    dr9 = dr9[mask]

    fj_pt = fj_pt[mask]
    fj_eta = fj_eta[mask]
    fj_phi = fj_phi[mask]
    fj_mass = fj_mass[mask]
    fj_sdmass = fj_sdmass[mask]
    fj_regmass = fj_regmass[mask]
    fj_nsub = fj_nsub[mask]
    fj_tau32 = fj_tau32[mask]
    fj_xbb = fj_xbb[mask]
    fj_xqq = fj_xqq[mask]
    fj_qcd = fj_qcd[mask]
    fj_higgs_idx = fj_higgs_idx[mask]

    signal = signal[mask]

    # lep   
    lep_pt = lep_pt[mask]
    lep_eta = lep_eta[mask]
    lep_phi = lep_phi[mask]

    # taus
    tau_pt = tau_pt[mask]
    tau_eta = tau_eta[mask]
    tau_phi = tau_phi[mask] 

    # global variables
    ht = events.ht[mask]
    met = events.met[mask]
    

    # mask to define zero-padded small-radius jets
    mask = pt > MIN_JET_PT
    mask_mass1 = mass1 > 20
    mask_mass2 = mass2 > 20
    mask_mass3 = mass3 > 20
    mask_mass4 = mass4 > 20
    mask_mass5 = mass5 > 20
    mask_mass6 = mass6 > 20
    mask_mass7 = mass7 > 20
    mask_mass8 = mass8 > 20
    mask_mass9 = mass9 > 20

    # mask to define zero-padded large-radius jets
    fj_mask = fj_pt > MIN_FJET_PT
    #fj_mask = fj_mass > 70

    # lep mask
    lep_mask = lep_pt > 25
    tau_mask = tau_pt > 20

    # require hadron_flavor == 5 (i.e. b-jet ghost association matching)
    higgs_idx = ak.where(higgs_idx != 0, ak.where(hadron_flavor == 5, higgs_idx, -1), 0)

    # index of small-radius jet if Higgs is reconstructed
    h1_bs = ak.local_index(higgs_idx)[higgs_idx == 1]
    h2_bs = ak.local_index(higgs_idx)[higgs_idx == 2]
    h3_bs = ak.local_index(higgs_idx)[higgs_idx == 3]

    # index of large-radius jet if Higgs is reconstructed
    h1_bb = ak.local_index(fj_higgs_idx)[fj_higgs_idx == 1]
    h2_bb = ak.local_index(fj_higgs_idx)[fj_higgs_idx == 2]
    h3_bb = ak.local_index(fj_higgs_idx)[fj_higgs_idx == 3]

    # check/fix small-radius jet truth (ensure max 2 small-radius jets per higgs)
    check = (
        np.unique(ak.count(h1_bs, axis=-1)).to_list()
        + np.unique(ak.count(h2_bs, axis=-1)).to_list()
        + np.unique(ak.count(h3_bs, axis=-1)).to_list()
    )
    if 3 in check:
        logging.warning("some Higgs bosons match to 3 small-radius jets! Check truth")

    # check/fix large-radius jet truth (ensure max 1 large-radius jet per higgs)
    fj_check = (
        np.unique(ak.count(h1_bb, axis=-1)).to_list()
        + np.unique(ak.count(h2_bb, axis=-1)).to_list()
        + np.unique(ak.count(h3_bb, axis=-1)).to_list()
    )
    if 2 in fj_check:
        logging.warning("some Higgs bosons match to 2 large-radius jets! Check truth")

    h1_bs = ak.fill_none(ak.pad_none(h1_bs, 2, clip=True), -1)
    h2_bs = ak.fill_none(ak.pad_none(h2_bs, 2, clip=True), -1)
    h3_bs = ak.fill_none(ak.pad_none(h3_bs, 2, clip=True), -1)

    h1_bb = ak.fill_none(ak.pad_none(h1_bb, 1, clip=True), -1)
    h2_bb = ak.fill_none(ak.pad_none(h2_bb, 1, clip=True), -1)
    h3_bb = ak.fill_none(ak.pad_none(h3_bb, 1, clip=True), -1)

    h1_b1, h1_b2 = h1_bs[:, 0], h1_bs[:, 1]
    h2_b1, h2_b2 = h2_bs[:, 0], h2_bs[:, 1]
    h3_b1, h3_b2 = h3_bs[:, 0], h3_bs[:, 1]

    # mask whether Higgs can be reconstructed as 2 small-radius jet
    h1_mask = ak.all(h1_bs != -1, axis=-1)
    h2_mask = ak.all(h2_bs != -1, axis=-1)
    h3_mask = ak.all(h3_bs != -1, axis=-1)

    # mask whether Higgs can be reconstructed as 1 large-radius jet
    h1_fj_mask = ak.all(h1_bb != -1, axis=-1)
    h2_fj_mask = ak.all(h2_bb != -1, axis=-1)
    h3_fj_mask = ak.all(h3_bb != -1, axis=-1)

    datasets = {}


    datasets["INPUTS/Jets/MASK"] = mask.to_numpy()
    datasets["INPUTS/Jets/pt"] = pt.to_numpy()
    datasets["INPUTS/Jets/ptcorr"] = ptcorr.to_numpy()
    datasets["INPUTS/Jets/eta"] = eta.to_numpy()
    datasets["INPUTS/Jets/phi"] = phi.to_numpy()
    datasets["INPUTS/Jets/sinphi"] = np.sin(phi.to_numpy())
    datasets["INPUTS/Jets/cosphi"] = np.cos(phi.to_numpy())
    datasets["INPUTS/Jets/btag"] = btag.to_numpy()
    #datasets["INPUTS/Jets/mass"] = mass.to_numpy()
    datasets["INPUTS/Jets/jetid"] = jet_id.to_numpy()
    datasets["INPUTS/Jets/matchedfj"] = matched_fj_idx.to_numpy()
    datasets["INPUTS/Jets/invmass"] = inv_mass.to_numpy()

    datasets["INPUTS/BoostedJets/MASK"] = fj_mask.to_numpy()
    datasets["INPUTS/BoostedJets/fj_pt"] = fj_pt.to_numpy()
    datasets["INPUTS/BoostedJets/fj_eta"] = fj_eta.to_numpy()
    datasets["INPUTS/BoostedJets/fj_phi"] = fj_phi.to_numpy()
    datasets["INPUTS/BoostedJets/fj_sinphi"] = np.sin(fj_phi.to_numpy())
    datasets["INPUTS/BoostedJets/fj_cosphi"] = np.cos(fj_phi.to_numpy())
    datasets["INPUTS/BoostedJets/fj_mass"] = fj_mass.to_numpy()
    datasets["INPUTS/BoostedJets/fj_sdmass"] = fj_sdmass.to_numpy()
    datasets["INPUTS/BoostedJets/fj_regmass"] = fj_regmass.to_numpy()
    datasets["INPUTS/BoostedJets/fj_nsub"] = fj_nsub.to_numpy()
    datasets["INPUTS/BoostedJets/fj_tau32"] = fj_tau32.to_numpy()
    datasets["INPUTS/BoostedJets/fj_xbb"] = fj_xbb.to_numpy()
    datasets["INPUTS/BoostedJets/fj_xqq"] = fj_xqq.to_numpy()
    datasets["INPUTS/BoostedJets/fj_qcd"] = fj_qcd.to_numpy()

    # Lep
    datasets["INPUTS/Leptons/MASK"] = lep_mask.to_numpy()
    datasets["INPUTS/Leptons/lep_pt"] = lep_pt.to_numpy()
    datasets["INPUTS/Leptons/lep_eta"] = lep_eta.to_numpy()
    datasets["INPUTS/Leptons/lep_phi"] = lep_phi.to_numpy()
    datasets["INPUTS/Leptons/lep_sinphi"] = np.sin(lep_phi.to_numpy())
    datasets["INPUTS/Leptons/lep_cosphi"] = np.cos(lep_phi.to_numpy())

    datasets["INPUTS/MET/met"] = met.to_numpy()
    datasets["INPUTS/HT/ht"] = ht.to_numpy()


    # Taus
    datasets["INPUTS/Taus/MASK"] = tau_mask.to_numpy()
    datasets["INPUTS/Taus/tau_pt"] = tau_pt.to_numpy()
    datasets["INPUTS/Taus/tau_eta"] = tau_eta.to_numpy()
    datasets["INPUTS/Taus/tau_phi"] = tau_phi.to_numpy()
    datasets["INPUTS/Taus/tau_sinphi"] = np.sin(tau_phi.to_numpy())
    datasets["INPUTS/Taus/tau_cosphi"] = np.cos(tau_phi.to_numpy())

    #for i in range(0, N_MASSES):
    #    datasets[f"INPUTS/Masses/MASK{i}"] = mask_mass.to_numpy()[:, i]
    #    datasets[f"INPUTS/Masses/mass{i}"] = mass.to_numpy()[:, i]

    # Higgses jets pair from AK4
    datasets[f"INPUTS/Jet1/MASK"] = mask_mass1.to_numpy()
    datasets[f"INPUTS/Jet1/mass1"] = mass1.to_numpy()
    datasets[f"INPUTS/Jet1/pt1"] = pt1.to_numpy()
    datasets[f"INPUTS/Jet1/eta1"] = eta1.to_numpy()
    datasets[f"INPUTS/Jet1/phi1"] = phi1.to_numpy()
    datasets[f"INPUTS/Jet1/sinphi1"] = np.sin(phi1.to_numpy())
    datasets[f"INPUTS/Jet1/cosphi1"] = np.cos(phi1.to_numpy())
    datasets[f"INPUTS/Jet1/dr1"] = dr1.to_numpy()

    datasets[f"INPUTS/Jet2/MASK"] = mask_mass2.to_numpy()
    datasets[f"INPUTS/Jet2/mass2"] = mass2.to_numpy()
    datasets[f"INPUTS/Jet2/pt2"] = pt2.to_numpy()
    datasets[f"INPUTS/Jet2/eta2"] = eta2.to_numpy()
    datasets[f"INPUTS/Jet2/phi2"] = phi2.to_numpy()
    datasets[f"INPUTS/Jet2/sinphi2"] = np.sin(phi2.to_numpy())
    datasets[f"INPUTS/Jet2/cosphi2"] = np.cos(phi2.to_numpy())
    datasets[f"INPUTS/Jet2/dr2"] = dr2.to_numpy()

    datasets[f"INPUTS/Jet3/MASK"] = mask_mass3.to_numpy()
    datasets[f"INPUTS/Jet3/mass3"] = mass3.to_numpy()
    datasets[f"INPUTS/Jet3/pt3"] = pt3.to_numpy()
    datasets[f"INPUTS/Jet3/eta3"] = eta3.to_numpy()
    datasets[f"INPUTS/Jet3/phi3"] = phi3.to_numpy()
    datasets[f"INPUTS/Jet3/sinphi3"] = np.sin(phi3.to_numpy())
    datasets[f"INPUTS/Jet3/cosphi3"] = np.cos(phi3.to_numpy())
    datasets[f"INPUTS/Jet3/dr3"] = dr3.to_numpy()

    datasets[f"INPUTS/Jet4/MASK"] = mask_mass4.to_numpy()
    datasets[f"INPUTS/Jet4/mass4"] = mass4.to_numpy()
    datasets[f"INPUTS/Jet4/pt4"] = pt4.to_numpy()
    datasets[f"INPUTS/Jet4/eta4"] = eta4.to_numpy()
    datasets[f"INPUTS/Jet4/phi4"] = phi4.to_numpy()
    datasets[f"INPUTS/Jet4/sinphi4"] = np.sin(phi4.to_numpy())
    datasets[f"INPUTS/Jet4/cosphi4"] = np.cos(phi4.to_numpy())
    datasets[f"INPUTS/Jet4/dr4"] = dr4.to_numpy()

    datasets[f"INPUTS/Jet5/MASK"] = mask_mass5.to_numpy()
    datasets[f"INPUTS/Jet5/mass5"] = mass5.to_numpy()
    datasets[f"INPUTS/Jet5/pt5"] = pt5.to_numpy()
    datasets[f"INPUTS/Jet5/eta5"] = eta5.to_numpy()
    datasets[f"INPUTS/Jet5/phi5"] = phi5.to_numpy()
    datasets[f"INPUTS/Jet5/sinphi5"] = np.sin(phi5.to_numpy())
    datasets[f"INPUTS/Jet5/cosphi5"] = np.cos(phi5.to_numpy())
    datasets[f"INPUTS/Jet5/dr5"] = dr5.to_numpy()

    datasets[f"INPUTS/Jet6/MASK"] = mask_mass6.to_numpy()
    datasets[f"INPUTS/Jet6/mass6"] = mass6.to_numpy()
    datasets[f"INPUTS/Jet6/pt6"] = pt6.to_numpy()
    datasets[f"INPUTS/Jet6/eta6"] = eta6.to_numpy()
    datasets[f"INPUTS/Jet6/phi6"] = phi6.to_numpy()
    datasets[f"INPUTS/Jet6/sinphi6"] = np.sin(phi6.to_numpy())
    datasets[f"INPUTS/Jet6/cosphi6"] = np.cos(phi6.to_numpy())
    datasets[f"INPUTS/Jet6/dr6"] = dr6.to_numpy()

    datasets[f"INPUTS/Jet7/MASK"] = mask_mass7.to_numpy()
    datasets[f"INPUTS/Jet7/mass7"] = mass7.to_numpy()
    datasets[f"INPUTS/Jet7/pt7"] = pt7.to_numpy()
    datasets[f"INPUTS/Jet7/eta7"] = eta7.to_numpy()
    datasets[f"INPUTS/Jet7/phi7"] = phi7.to_numpy()
    datasets[f"INPUTS/Jet7/sinphi7"] = np.sin(phi7.to_numpy())
    datasets[f"INPUTS/Jet7/cosphi7"] = np.cos(phi7.to_numpy())
    datasets[f"INPUTS/Jet7/dr7"] = dr7.to_numpy()

    datasets[f"INPUTS/Jet8/MASK"] = mask_mass8.to_numpy()
    datasets[f"INPUTS/Jet8/mass8"] = mass8.to_numpy()
    datasets[f"INPUTS/Jet8/pt8"] = pt8.to_numpy()
    datasets[f"INPUTS/Jet8/eta8"] = eta8.to_numpy()
    datasets[f"INPUTS/Jet8/phi8"] = phi8.to_numpy()
    datasets[f"INPUTS/Jet8/sinphi8"] = np.sin(phi8.to_numpy())
    datasets[f"INPUTS/Jet8/cosphi8"] = np.cos(phi8.to_numpy())
    datasets[f"INPUTS/Jet8/dr8"] = dr8.to_numpy()

    datasets[f"INPUTS/Jet9/MASK"] = mask_mass9.to_numpy()
    datasets[f"INPUTS/Jet9/mass9"] = mass9.to_numpy()
    datasets[f"INPUTS/Jet9/pt9"] = pt9.to_numpy()
    datasets[f"INPUTS/Jet9/eta9"] = eta9.to_numpy()
    datasets[f"INPUTS/Jet9/phi9"] = phi9.to_numpy()
    datasets[f"INPUTS/Jet9/sinphi9"] = np.sin(phi9.to_numpy())
    datasets[f"INPUTS/Jet9/cosphi9"] = np.cos(phi9.to_numpy())
    datasets[f"INPUTS/Jet9/dr9"] = dr9.to_numpy()

    datasets["TARGETS/h1/mask"] = h1_mask.to_numpy()
    datasets["TARGETS/h1/b1"] = h1_b1.to_numpy()
    datasets["TARGETS/h1/b2"] = h1_b2.to_numpy()

    datasets["TARGETS/h2/mask"] = h2_mask.to_numpy()
    datasets["TARGETS/h2/b1"] = h2_b1.to_numpy()
    datasets["TARGETS/h2/b2"] = h2_b2.to_numpy()

    datasets["TARGETS/h3/mask"] = h3_mask.to_numpy()
    datasets["TARGETS/h3/b1"] = h3_b1.to_numpy()
    datasets["TARGETS/h3/b2"] = h3_b2.to_numpy()

    datasets["TARGETS/bh1/mask"] = h1_fj_mask.to_numpy()
    datasets["TARGETS/bh1/bb"] = h1_bb.to_numpy().reshape(h1_fj_mask.to_numpy().shape)

    datasets["TARGETS/bh2/mask"] = h2_fj_mask.to_numpy()
    datasets["TARGETS/bh2/bb"] = h2_bb.to_numpy().reshape(h2_fj_mask.to_numpy().shape)

    datasets["TARGETS/bh3/mask"] = h3_fj_mask.to_numpy()
    datasets["TARGETS/bh3/bb"] = h3_bb.to_numpy().reshape(h3_fj_mask.to_numpy().shape)

    datasets["CLASSIFICATIONS/EVENT/signal"] = signal.to_numpy()


    return datasets


@click.command()
@click.argument("in-files", nargs=-1)
@click.option("--out-file", default=f"{PROJECT_DIR}/data/cms/hhh_training.h5", help="Output file.")
@click.option("--train-frac", default=0.95, help="Fraction for training.")
def main(in_files, out_file, train_frac):
    all_datasets = {}
    for file_name in in_files:
        print(file_name)
        if 'JetHT' in file_name: continue
        if 'SingleMuon' in file_name: continue
        if 'BTagCSV' in file_name: continue

        with uproot.open(file_name) as in_file:
            num_entries = in_file["Events"].num_entries
            if "training" in out_file:
                entry_start = None
                entry_stop = int(train_frac * num_entries)
            else:
                entry_start = int(train_frac * num_entries)
                entry_stop = None
            events = NanoEventsFactory.from_root(
                in_file,
                treepath="Events",
                entry_start=entry_start,
                entry_stop=entry_stop,
                schemaclass=BaseSchema,
            ).events()
            #if 'GluGluToHHHTo6B_SM' in file_name:
            #    events.signal = 1
            #else:
            #    events.signal = 0
            for key,value in mappings.items():
                if key in file_name:
                    events.signal = value

            datasets = get_datasets(events)
            for dataset_name, data in datasets.items():
                if dataset_name not in all_datasets:
                    all_datasets[dataset_name] = []
                all_datasets[dataset_name].append(data)



    print(all_datasets.keys())
    with h5py.File(out_file, "w") as output:

        for dataset_name, all_data in all_datasets.items():
            concat_data = np.concatenate(all_data, axis=0)
            output.create_dataset(dataset_name, data=concat_data)


    # shuffle randomly
    data = h5py.File(out_file, 'r')


    with h5py.File(out_file.replace('.h5','_random.h5'), 'w') as out:
        print(data)
        indexes = np.arange(data['CLASSIFICATIONS/EVENT/signal'].shape[0])
        np.random.shuffle(indexes)
        for key in all_datasets.keys():
            print(key)
            feed = np.take(np.array(data[key]), indexes, axis=0)
            out.create_dataset(key, data=feed)


if __name__ == "__main__":
    main()