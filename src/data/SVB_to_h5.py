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
MIN_JET_PT = 20
MIN_JETS = 6
MIN_MASS = 50
PROJECT_DIR = Path(__file__).resolve().parents[3]


def get_n_features(name, events, iterator):
    if name.format(i=iterator[0]) not in dir(events):
        logging.warning(f"Variable {name.format(i=iterator[0])} does not exist in tree; returning all 0s")
        return ak.from_numpy(np.zeros((len(events), len(iterator))))
    return ak.concatenate(
        [np.expand_dims(events[name.format(i=i)], axis=-1) for i in iterator],
        axis=-1,
    )


def get_datasets(events, is_signal):
    # small-radius jet info
    pt = get_n_features("jet{i}Pt", events, range(1, N_JETS + 1))
    eta = get_n_features("jet{i}Eta", events, range(1, N_JETS + 1))
    phi = get_n_features("jet{i}Phi", events, range(1, N_JETS + 1))
    btag = get_n_features("jet{i}DeepFlavB", events, range(1, N_JETS + 1))
    mass = get_n_features("jet{i}Mass", events, range(1, N_JETS + 1))
    jet_id = get_n_features("jet{i}JetId", events, range(1, N_JETS + 1))
    higgs_idx = get_n_features("jet{i}HiggsMatchedIndex", events, range(1, N_JETS + 1))
    hadron_flavor = get_n_features("jet{i}HadronFlavour", events, range(1, N_JETS + 1))
    matched_fj_idx = get_n_features("jet{i}FatJetMatchedIndex", events, range(1, N_JETS + 1))
    

    # keep events with >= MIN_JETS small-radius jets
    mask = ak.num(pt[pt > MIN_JET_PT]) >= MIN_JETS
    pt = pt[mask]
    eta = eta[mask]
    phi = phi[mask]
    btag = btag[mask]
    mass = mass[mask]
    jet_id = jet_id[mask]
    higgs_idx = higgs_idx[mask]
    hadron_flavor = hadron_flavor[mask]
    matched_fj_idx = matched_fj_idx[mask]

    # mask to define zero-padded small-radius jets
    mask = pt > MIN_JET_PT
    mask_mass = mass > MIN_MASS

    # require hadron_flavor == 5 (i.e. b-jet ghost association matching)
    higgs_idx = ak.where(higgs_idx != 0, ak.where(hadron_flavor == 5, higgs_idx, -1), 0)

    # index of small-radius jet if Higgs is reconstructed
    h1_bs = ak.local_index(higgs_idx)[higgs_idx == 1]
    h2_bs = ak.local_index(higgs_idx)[higgs_idx == 2]
    h3_bs = ak.local_index(higgs_idx)[higgs_idx == 3]

    # check/fix small-radius jet truth (ensure max 2 small-radius jets per higgs)
    check = (
        np.unique(ak.count(h1_bs, axis=-1)).to_list()
        + np.unique(ak.count(h2_bs, axis=-1)).to_list()
        + np.unique(ak.count(h3_bs, axis=-1)).to_list()
    )
    if 3 in check:
        logging.warning("some Higgs bosons match to 3 small-radius jets! Check truth")


    h1_bs = ak.fill_none(ak.pad_none(h1_bs, 2, clip=True), -1)
    h2_bs = ak.fill_none(ak.pad_none(h2_bs, 2, clip=True), -1)
    h3_bs = ak.fill_none(ak.pad_none(h3_bs, 2, clip=True), -1)

    h1_b1, h1_b2 = h1_bs[:, 0], h1_bs[:, 1]
    h2_b1, h2_b2 = h2_bs[:, 0], h2_bs[:, 1]
    h3_b1, h3_b2 = h3_bs[:, 0], h3_bs[:, 1]

    # mask whether Higgs can be reconstructed as 2 small-radius jet
    h1_mask = ak.all(h1_bs != -1, axis=-1)
    h2_mask = ak.all(h2_bs != -1, axis=-1)
    h3_mask = ak.all(h3_bs != -1, axis=-1)

    datasets = {}
    datasets["INPUTS/Jets/MASK"] = mask.to_numpy()
    datasets["INPUTS/Jets/pt"] = pt.to_numpy()
    datasets["INPUTS/Jets/eta"] = eta.to_numpy()
    datasets["INPUTS/Jets/phi"] = phi.to_numpy()
    datasets["INPUTS/Jets/sinphi"] = np.sin(phi.to_numpy())
    datasets["INPUTS/Jets/cosphi"] = np.cos(phi.to_numpy())
    datasets["INPUTS/Jets/btag"] = btag.to_numpy()
    datasets["INPUTS/Jets/mass"] = mass.to_numpy()
    datasets["INPUTS/Jets/jetid"] = jet_id.to_numpy()
    datasets["INPUTS/Jets/matchedfj"] = matched_fj_idx.to_numpy()

    datasets["TARGETS/h1/mask"] = h1_mask.to_numpy()
    datasets["TARGETS/h1/b1"] = h1_b1.to_numpy()
    datasets["TARGETS/h1/b2"] = h1_b2.to_numpy()

    datasets["TARGETS/h2/mask"] = h2_mask.to_numpy()
    datasets["TARGETS/h2/b1"] = h2_b1.to_numpy()
    datasets["TARGETS/h2/b2"] = h2_b2.to_numpy()

    datasets["TARGETS/h3/mask"] = h3_mask.to_numpy()
    datasets["TARGETS/h3/b1"] = h3_b1.to_numpy()
    datasets["TARGETS/h3/b2"] = h3_b2.to_numpy()

    # Add label for signal vs background classification task
    datasets["CLASSIFICATIONS/EVENT/signal"] = (np.ones(len(pt)) if is_signal else np.zeros(len(pt))).astype(int)

    # Add label for categorization
    datasets["CLASSIFICATIONS/EVENT/category"] = h1_mask.to_numpy().astype(int) + h2_mask.to_numpy().astype(int) + h3_mask.to_numpy().astype(int)
    
    return datasets


@click.command()
@click.argument("signal-file", nargs=1)
@click.argument("background-file", nargs=1)
@click.option("--out-file", default=f"{PROJECT_DIR}/data/cms/hhh_training.h5", help="Output file.")
@click.option("--train-frac", default=0.95, help="Fraction for training.")
def main(signal_file, background_file, out_file, train_frac):
    all_datasets = {}
    for file_name, is_signal in [(signal_file, True), (background_file, False)]:
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

            datasets = get_datasets(events, is_signal)
            for dataset_name, data in datasets.items():
                if dataset_name not in all_datasets:
                    all_datasets[dataset_name] = []
                all_datasets[dataset_name].append(data)

    with h5py.File(out_file, "w") as output:
        for dataset_name, all_data in all_datasets.items():
            concat_data = np.concatenate(all_data, axis=0)
            output.create_dataset(dataset_name, data=concat_data)


if __name__ == "__main__":
    main()