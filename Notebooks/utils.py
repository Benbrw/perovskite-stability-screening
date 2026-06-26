# Before we start, Let's import all the necessary libraries
from mp_api.client import MPRester
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Section 1: Global Variables used for each notebook: 

# Ionic radii for common elements in perovskite structures.
# We hardcode these values as they are well established in literature
# Aditionally, we deal with ions whereas the magpie features deal with neutral atoms
ionic_radii = {
    "Cs": 1.88,
    "MA": 2.17,
    "FA": 2.53,
    "Rb": 1.72,
    "K": 1.64,
    "Pb": 1.19,
    "Sn": 1.10,
    "Ge": 0.73,
    "I": 2.20,
    "Br": 1.96,
    "Cl": 1.81,
    "Ca": 1.34,
    "Na": 1.02,
    "Ag": 1.15,
    "Cu": 0.73,
    }

halides = {"I", "Br", "Cl"}

metals = {"Pb", "Sn", "Ge"}


# Section 2: Helper Functons needed for each notebook:


# Checking Candidates Plausibility of Perovskite Structure
def is_plausible_singleB_perovskite(formula_str):
    try:
        from pymatgen.core import Composition
           
        comp = Composition(formula_str)
        elements = [str(e) for e in comp.elements]

        # Currently restricting to single B-site perovskites.
        # These are the most common and well-studied
        # We want to avoid the complexity of double perovskites which can have more than 4 elements.
        has_one_metal = sum(1 for e in elements if e in metals) == 1
        has_one_halide = sum(1 for e in elements if e in halides) >= 1
        has_a_site = any(e not in metals and e not in halides for e in elements)
        not_too_complex = len(elements) <= 4
        return has_one_metal and has_one_halide and has_a_site and not_too_complex
    except:
        return False

# Obtain our training and testing data from the Materials Project Database
def query_materials_project(API_KEY, Metal_Halide_Pairings, number_of_elements=None):
    data = []
    with MPRester(API_KEY) as mpr:
        for pairing in Metal_Halide_Pairings:
            search_params = {
                "elements": pairing,
                "band_gap": (0.5, 3.0),
                "fields": [
                    "formula_pretty",
                    "band_gap",
                    "energy_above_hull",
                    "formation_energy_per_atom",
                    "volume",
                    "density",
                    "nsites",
                    "symmetry"
                ]
            }
            # Add the number of elements to the search parameters if specified
            # This is primarily used for restricting the search to materials with the ABX3 structure
            # Number of elements should be 3 for ABX3 structures
            if number_of_elements is not None:
                search_params["num_elements"] = number_of_elements

            docs = mpr.materials.summary.search(**search_params)
            for doc in docs:
                sym = getattr(doc, "symmetry", None)
                data.append({
                    "formula": doc.formula_pretty,
                    "band_gap": getattr(doc, "band_gap", None),
                    "energy_above_hull": getattr(doc, "energy_above_hull", None),
                    "formation_energy": getattr(doc, "formation_energy_per_atom", None),
                    "volume": getattr(doc, "volume", None),
                    "density": getattr(doc, "density", None),
                    "nsites": getattr(doc, "nsites", None), 
                    "spacegroup_number": sym.number if sym is not None else None
                })
    df = pd.DataFrame(data)
    df = df.dropna(subset=["band_gap", "energy_above_hull"])
    df = df[df["formula"].apply(is_plausible_singleB_perovskite)]
    return df

# Computing the Goldschmidt Tolerance Factor 
def compute_tolerance_factor(composition):
    # Attempt to calculate the Goldschmidt Tolerance Factor
    try:
        elements = list(composition)
        for element in elements:
            if str(element) in halides:
                X = str(element)
        
        for element in elements:
            if str(element) in metals:
                B = str(element)   

        for element in elements:
            if str(element) not in halides and str(element) not in metals:
                A = str(element)
            
        Arad = ionic_radii[A]
        Brad = ionic_radii[B]
        Xrad = ionic_radii[X]
        gtf = (Arad + Xrad)/(np.sqrt(2*(Brad + Xrad)))

        return gtf
    except:
        # If any error occurs, just return NaN
        return np.nan

# For training purposes we do not want to omit different entries of the same formula. In fact they are more valuable
# To make our data comparison at the end make sense, we must compress the features to have a single values correspondign to a single formula
