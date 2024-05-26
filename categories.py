from typing import Dict, List, Iterator, Any
import pandas as pd


class Categories:
    """
    This class is used to store the categories of the dataset.
    It maps the categories from full names to the corresponding names used in the dataset.
    Example category mapping:
    categories_lookup_dict = {
            'Atrial fibrillation': ['Atrial fibrillation'],
            'Atrial flutter': ['Atrial flutter'],
            'Atrial premature complex(es) - APC APB': ['Atrial premature complex(es)',
                                                       'Atrial premature complexes_ nonconducted'],
                                                       }
    """
    def __init__(self, categories_lookup_dict: Dict[str, List[str]]):
        self.categories_lookup_dict = categories_lookup_dict
        self.categories = sorted(categories_lookup_dict.keys())
        self.num_categories = len(self.categories)
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}

    def __len__(self) -> int:
        return self.num_categories

    def __getitem__(self, idx: int) -> str:
        return self.categories[idx]

    def __iter__(self) -> Iterator[str]:
        return iter(self.categories)

    def __contains__(self, item: str) -> bool:
        return item in self.categories

    def __repr__(self) -> str:
        return f"Categories: {self.categories}"

    def __eq__(self, other: Any) -> bool:
        return (
            self.categories_lookup_dict == other.categories_lookup_dict
            if isinstance(other, Categories)
            else False
        )

    def __hash__(self) -> int:
        return hash(self.categories_lookup_dict)

    def get_category_idx(self, category: str) -> int:
        """
        Get the index of a category.
        Args:
            category: the category to get the index of.
        Returns:
            The index of the category.
        """
        return self.category_to_idx[category]

    def get_category(self, idx: int) -> str:
        """
        Get the category of an index.
        Args:
            idx: the index to get the category of.
        Returns:
            The category of the index.
        """
        return self.categories[idx]

    def get_categories(self, idxs: List[int]) -> List[str]:
        """
        Get the categories of a list of indices.
        Args:
            idxs: the indices to get the categories of.
        Returns:
            The categories of the indices.
        """
        return [self.categories[idx] for idx in idxs]

    def get_categories_lookup_dict(self) -> Dict[str, List[str]]:
        """
        Get the category mapping.
        Returns:
            The category mapping.
        """
        return self.categories_lookup_dict

    def get_categories_lookup_dict_inverse(self) -> Dict[str, str]:
        """
        Get the inverse category mapping.
        Returns:
            The inverse category mapping.
        """
        return {cat: cat_full for cat_full, cats in self.categories_lookup_dict.items() for cat in cats}


ny_categories_lookup_dict = \
    {
        'AV Block - First-degree': ['AV Block', ' type I'],
        'AV Block - Second-degree': ['AV Block - Second-degree',
                                     ' Mobitz type II',
                                     ' Mobitz type I (Wenckebach)'], #                                     ' type I'
        'AV Block - Third-degree (Complete)': ' complete (third-degree)',
        'Abnormal P-wave Axis': 'Abnormal P-wave axis',
        'Acute Pericarditis': 'Acute pericarditis',
        'Atrial Fibrillation': 'Atrial fibrillation',
        'Atrial Fibrillation & Flutter': ['Atrial fibrillation', 'Atrial flutter'],
        'Atrial Flutter': 'Atrial flutter',
        'Atrial Premature Complex(es) - APC APB': ['Atrial premature complex(es) - APC APB',
                                                   'Ectopic atrial rhythm',
                                                   'Ectopic atrial tachycardia'],
        'Atrial Tachycardia': 'Ectopic atrial tachycardia',
        'Clockwise Or Counterclockwise Vectorcardiographic Loop': 'Clockwise or counterclockwise vectorcardiographic loop',
        'Complete Heart Block': 'complete (third-degree)',
        'Complete Left Bundle Branch Block': 'Bundle Branch Block - Left - LBBB',
        'Complete Right Bundle Branch Block': 'Bundle Branch Block - Right - RBBB',
        'Digitalis Effect': 'Digitalis effect',
        'Early Repolarization': 'Early repolarization',
        'Ectopic Atrial Tachycardia': 'Ectopic atrial tachycardia',
        'Electrode Reversal': 'Extremity electrode reversal',
        'Fusion Beats': 'Fusion complex(es)',
        'Incomplete Right Bundle Branch Block': 'Bundle-branch block - RBBB - incomplete',
        'Intraventricular Conduction Delay': ['Intraventricular conduction delay',
                                                'Wide-QRS tachycardia'],
        'Ischemia': ['Ischemia','Ischemia - location unspecified'],
        'Junctional Escape': 'Junctional escape complex(es)',
        'Junctional Tachycardia': 'Junctional tachycardia',
        'Left Anterior Fascicular Block': 'Left anterior fascicular block',
        'Left Atrial Enlargement': 'Left atrial enlargement',
        'Left Axis Deviation': 'Left-axis deviation',
        'Left Bundle Branch Block': 'Bundle Branch Block - Left - LBBB',
        'Left Posterior Fascicular Block': 'Left posterior fascicular block',
        'Left Ventricular Hypertrophy': ['Left ventricular hypertrophy',
                                         'ST-T change due to ventricular hypertrophy',
                                         'Biventricular hypertrophy'],
        'Low QRS Voltages': 'Low voltage',
        'Myocardial Infarction': ['STEMI - Anteroseptal',
                                  'STEMI - Anterior',
                                  'STEMI - Inferior or Inferolateral',
                                  'STEMI - Lateral',
                                  'STEMI - Posterior',
                                  'STEMI - Right Ventricular'],
        'Myocardial Infarction - Anterior': ['STEMI - Anterior','STEMI - Anteroseptal'],
        'Myocardial Infarction - Anteroseptal': 'STEMI - Anteroseptal',
        'Myocardial Infarction - Inferior Or Inferolateral': 'STEMI - Inferior or Inferolateral',
        'Myocardial Infarction - Lateral': 'STEMI - Lateral',
        'Myocardial Infarction - Posterior': 'STEMI - Posterior',
        'Myocardial Infarction - Right Ventricular': 'STEMI - Right Ventricular',
        'Myocardial Infarction - Septal': ['STEMI - Anteroseptal'],
        'Myocardial Ischemia': 'Ischemia',
        'Nonspecific Intraventricular Conduction Disorder': 'Intraventricular conduction delay',
        'Normal Sinus Rhythm': 'Normal ECG',
        'Normal Variant': ['Normal variant'],  #                           'Sinus arrhythmia'
        'PR Interval - Prolonged': ['PR Interval - Prolonged',
                                    'AV Block',
                                     ' type I'],
        'PR Interval - Short': ['PR Interval - Short',
                                'Wolff-Parkinson-White'],
        'Pacing': ['Pacing - Ventricular-paced complex(es) or rhythm',
                   'Pacing - Atrial-sensed ventricular-paced complex(es) or rhythm',
                   'Pacing - AV dual-paced complex(es) or rhythm',
                   'Pacing - Atrial-paced complex(es) or rhythm',
                   'Pacing Ventricular pacing',
                   'Pacing - Failure to pace',
                   'Pacing - Failure to inhibit',
                   'Pacing - Failure to capture'],
        'Poor R Wave Progression': 'Abnormal precordial R-wave progression',
        'Premature Atrial Contractions': ['Atrial premature complex(es) - APC APB',
                                            'Ectopic atrial rhythm',
                                            'Ectopic atrial tachycardia'],                   
        'Premature Ventricular Contractions': ['Ventricular premature complex(es) - VPB - VPC',
                                               'Ventricular preexcitation'],
        'Pulmonary Disease': 'Pulmonary disease',
        'QRS - Prolonged': 'Wide-QRS tachycardia',
        'QT Interval - Prolonged': 'QT Interval - Prolonged',
        'QT Interval - Short': 'QT Interval - Short',
        'Q Wave Abnormal': 'Q wave abnormal',
        'Right Atrial Enlargement': 'Right atrial enlargement',
        'Right Axis Deviation': ['Right-axis deviation','Left posterior fascicular block','Right superior axis','Pulmonary disease'],
        'Right Bundle Branch Block - General': ['Bundle Branch Block - Right - RBBB',                                                
                                                'Bundle-branch block - RBBB - incomplete'],
        'Right Superior Axis': 'Right superior axis',
        'Right Ventricular Hypertrophy': ['Right ventricular hypertrophy',
                                          'Biventricular hypertrophy'],
        'ST Changes': ['ST changes - Nonspecific T-wave abnormality',
                       'ST changes - Nonspecific ST deviation',
                       'ST changes - Nonspecific ST deviation with T-wave change',
                       'ST-T change due to ventricular hypertrophy'],
        'ST Changes - Nonspecific ST Deviation': 'ST changes - Nonspecific ST deviation',
        'ST Changes - Nonspecific ST Deviation With T-wave Change': 'ST changes - Nonspecific ST deviation with T-wave change',
        'ST Changes - Nonspecific T-wave Abnormality': 'ST changes - Nonspecific T-wave abnormality',
        'ST Elevation': ['STEMI - Anteroseptal',
                    'STEMI - Anterior',
                    'STEMI - Inferior or Inferolateral',
                    'STEMI - Lateral',
                    'STEMI - Posterior',
                    'STEMI - Right Ventricular'],
        'STEMI': ['STEMI - Anteroseptal',
                    'STEMI - Anterior',
                    'STEMI - Inferior or Inferolateral',
                    'STEMI - Lateral',
                    'STEMI - Posterior',
                    'STEMI - Right Ventricular'],
        'Sinosatrial Block': 'Sinosatrial block',
        'Sinus Arrhythmia': 'Sinus arrhythmia',
        'Sinus Bradycardia': 'Sinus bradycardia',
        'Sinus Tachycardia': 'Sinus tachycardia',
        'Supraventricular Tachycardia': ['Supraventricular tachycardia',
                                         'Junctional tachycardia'],
        'T Wave Abnormal': ['ST-T change due to ventricular hypertrophy',
                            'ST changes - Nonspecific T-wave abnormality',
                            'ST changes - Nonspecific ST deviation with T-wave change'],
        'TU Fusion': 'TU fusion',
        'U Wave Abnormal':'Prominent U waves',
        'Ventricular Escape Rhythm': 'Ventricular escape complex(es)',
        'Ventricular Pre Excitation': 'Ventricular preexcitation',
        'Ventricular Tachycardia': 'Ventricular tachycardia',
        'Wandering Atrial Pacemaker': 'Wandering atrial pacemaker',
        'Wide-QRS Tachycardia': ['Wide-QRS tachycardia',
                                'Ventricular tachycardia'],
        'Wolff-Parkinson-White': 'Wolff-Parkinson-White'
    }                               

Mobile_Labeled_categories_lookup_dict = {
        'Aberrant Conduction of Supraventricular Beat(s)': 'Aberrant Conduction of Supraventricular Beat(s)',
        'Abnormal P-wave Axis': 'Abnormal P-wave axis',
        'Accelerated Idioventricular Rhythm': 'Accelerated Idioventricular Rhythm',
        'Accelerated Junctional Rhythm': 'Accelerated Junctional Rhythm',
        'Atrial Pacing': 'Atrial Pacing',
        'AV Block - First-degree': 'AV Block - First-degree',
        'AV Block - Second-degree': 'AV Block - Second-degree',
        'AV Block - Third-degree (Complete)': ['AV Block - Third-degree (Complete)',
                                               'Complete Heart Block'],
        'AV Dissociation': 'AV Dissociation',
        'AV Junctional Rhythm': ['AV Junctional Rhythm',
                                 'Junctional Escape',
                                 'Junctional Tachycardia',
                                 'AV Node Reentrant Tachycardia (AVNRT)',
                                 'Accelerated Junctional Rhythm'],
        'Abnormal P-wave Axis': 'Abnormal P-wave axis',
        'Acute Myocardial Ischemia': 'Acute Myocardial Ischemia',
        'Acute Pericarditis': 'Acute Pericarditis',
        'Anterior Ischemia': 'Anterior Ischemia',
        'Anterolateral Ischemia': 'Anterolateral Ischemia',
        'Atrial Bigeminy': 'Atrial Bigeminy',
        'Atrial Fibrillation': ['Atrial fibrillation','Atrial Fibrillation with a Rapid Ventricular Rate','Atrial Fibrillation with a Slow Ventricular Rate'],
        'Atrial Fibrillation & Flutter': ['Atrial fibrillation','Atrial flutter','Typical Atrial Flutter','Atypical Atrial Flutter',
                                          'Atrial Fibrillation with a Rapid Ventricular Rate',
                                          'Atrial Fibrillation with a Slow Ventricular Rate'],
        'Atrial Flutter': ['Atrial flutter',
                           'Typical Atrial Flutter',
                            'Atypical Atrial Flutter'],
        'Atrial Hypertrophy': 'Atrial Hypertrophy',
        'Atrial Premature Complex(es) - APC APB': ['Atrial premature complex(es) - APC APB',
                                                   'Ectopic atrial rhythm',
                                                   'Atrial Bigeminy'],
        'Atrial Rhythm': ['Atrial Rhythm','Accelerated Atrial Escape Rhythm'],
        'Atrial Tachycardia': ['Atrial Tachycardia',
                               'Ectopic Atrial Tachycardia'],
        'Bifascicular Block': 'Bifascicular block',
        'Clockwise Or Counterclockwise Vectorcardiographic Loop': ['Clockwise Rotation',
                                                                    'Counterclockwise Rotation'],
        'Clockwise Rotation': 'Clockwise Rotation',
        'Complete Heart Block': ['complete (third-degree)',
                                'Complete Heart Block',
                                'AV Block - Third-degree (Complete)'],
        'Complete Left Bundle Branch Block':  ['Left Bundle Branch Block - General',
                                                'Complete Left Bundle Branch Block'],
        'Complete Right Bundle Branch Block': ['Bundle Branch Block - Right - RBBB',
                                               'Complete Right Bundle Branch Block',
                                               'Right Bundle Branch Block - General'],
        'Counterclockwise Rotation': 'Counterclockwise Rotation',             
        'Dextrocardia': 'Dextrocardia',
        'De Winter T Wave': 'De Winter T Wave',
        'Digitalis Effect': 'Digitalis Effect',
        'Diffuse Intraventricular Block': 'Diffuse Intraventricular Block',
        'Early Repolarization': 'Early repolarization',
        'Early Q wave Formation': 'Early Q wave formation',
        'Ectopic Atrial Tachycardia': 'Ectopic Atrial Tachycardia',
        'Electrode Reversal': ['Electrode Reversal','Lead Displacement'],
        'Escape Bigeminy': 'Escape Bigeminy',
        'Fusion Beats': 'Fusion Beats',
        'Hyperacute T waves': 'Hyperacute T waves',
        'Idioventricular Rhythm': 'Idioventricular Rhythm',
        'Incomplete Left Bundle Branch Block': ['Incomplete Left Bundle Branch Block',
                                                'Intermittent Left Bundle Branch Block'],
        'Incomplete Right Bundle Branch Block': 'Incomplete Right Bundle Branch Block',
        'Inferior Ischaemia': 'Inferior Ischaemia',
        'Inferior ST Segment Depression': 'Inferior ST Segment Depression',
        'Infero-posterior STEMI': 'Infero-posterior STEMI',
        'Intraventricular Conduction Delay': ['Intraventricular Conduction Delay',
                                            'Wide Complex Tachycardia',                                              
                                            'QRS - Prolonged',
                                            'Poor R Wave Progression',
                                            'Idioventricular Rhythm'],
        'Ischemia': ['Myocardial Ischemia',
                                'Anterior Ischemia'
                                'Inferior Ischemia',
                                'Inferior Ischaemia',
                                'Lateral Ischemia',
                                'Ischemia',
                                'Acute Myocardial Ischemia',
                                'Anterolateral Ischemia'],
        'Junctional Escape': 'Junctional Escape',
        'Junctional Tachycardia': ['Junctional Tachycardia','AV Node Reentrant Tachycardia (AVNRT)'],
        'J Wave (Osborn Wave)': 'J Wave (Osborn Wave)',
        'Lateral Ischemia': 'Lateral Ischemia',
        'Left Anterior Fascicular Block': 'Left anterior fascicular block',
        'Left Atrial Enlargement': ['Left Atrial Enlargement',
                                    'Left Atrial Hypertrophy'],
        'Left Axis Deviation': ['Left axis deviation',
                                'Left anterior fascicular block'],
        'Left Bundle Branch Block': ['Left Bundle Branch Block - General',
                                     'Incomplete Left Bundle Branch Block',
                                     'Complete Left Bundle Branch Block',
                                     'Intermittent Left Bundle Branch Block'],
        'Left Posterior Fascicular Block': ['Left Posterior Fascicular Block','Bifascicular Block'],
        'Left Ventricular Hypertrophy': ['Left Ventricular Hypertrophy','Takotsubo Cardiomyopathy'],
        'Low QRS Voltages': 'Low QRS Voltages',
        'Myocardial Infarction': ['Myocardial Infarction - Inferior Or Inferolateral',
                                'Myocardial Infarction - Right Ventricular',
                                'Myocardial Infarction - Anterior',
                                'Myocardial Infarction',
                                'Myocardial Infarction - Anteroseptal',
                                'Myocardial Infarction - Posterior',
                                'Myocardial Infarction - Lateral',
                                'Myocardial Infarction - Right Ventricular',                                  
                                'Infero-posterior STEMI',
                                'Septal Infract',
                                'STEMI - Inferior or Inferolateral',
                                'STEMI - Lateral',
                                'Septal Myocardial Infraction'],
        'Myocardial Infarction - Anterior': ['STEMI - Anterior','STEMI - Anteroseptal'],
        'Myocardial Infarction - Anteroseptal': 'STEMI - Anteroseptal',
        'Myocardial Infarction - Inferior Or Inferolateral': ['Myocardial Infarction - Inferior Or Inferolateral',
                                                              'STEMI - Inferior or Inferolateral'],
        'Myocardial Infarction - Lateral': 'Myocardial Infarction - Lateral',
        'Myocardial Infarction - Posterior': 'Myocardial Infarction - Posterior',
        'Myocardial Infarction - Right Ventricular': 'STEMI - Right Ventricular',
        'Myocardial Infarction - Right Ventricular': 'Myocardial Infarction - Right Ventricular',
        'Myocardial Infarction - Septal': ['Myocardial Infarction - Septal',
                                           'Myocardial Infarction - Anteroseptal',
                                           'Septal Infract',
                                           'Septal Myocardial Infraction'],
        'Myocardial Ischemia': ['Ischemia',
                                'Anterior Ischemia',
                                'Inferior Ischaemia',
                                'Lateral Ischemia',
                                'Ischemia'],
        'Narrow QRS Complex': 'Narrow QRS Complex',
        'Nonspecific Intraventricular Conduction Disorder': ['Nonspecific Intraventricular Conduction Disorder',
                                                             'Nonspecific Intraventricular Block'],
        'Normal Sinus Rhythm': 'Normal ECG',
        'Normal Variant': ['Normal Variant',
                        'Sinus arrhythmia'],
        'Northwest Axis': 'Northwest Axis',
        'Pacing': ['Pacing - Ventricular-paced complex(es) or rhythm',
                   'Pacing - Atrial-sensed ventricular-paced complex(es) or rhythm',
                   'Pacing - AV dual-paced complex(es) or rhythm',
                   'Pacing - Atrial-paced complex(es) or rhythm',
                   'Pacing Ventricular pacing',
                   'Pacing - Failure to pace',
                   'Pacing - Failure to inhibit',
                   'Pacing - Failure to capture',
                   'Pacing',
                   'Ventricular Pacing',
                   'Atrial Pacing',
                   'Arial Pacing'],
        'Poor R Wave Progression': 'Poor R Wave Progression',
        'PR Interval - Prolonged': ['PR Interval - Prolonged',
                                    'AV Block - First-degree'],
        'PR Interval - Short': ['PR Interval - Short',  
                                'Wolff-Parkinson-White',
                                ],      #'AV Junctional Rhythm'
        'Premature Atrial Contractions': ['Atrial Premature Complex(es) - APC APB',
                                            'Ectopic Atrial Tachycardia',
                                            'Atrial Bigeminy'],
        'Premature Junctional Contractions': ['Junctional Premature Complex'],
        'Premature Ventricular Contractions': ['Premature Ventricular Contractions',
                                                'Ventricular Bigeminy',
                                                'Ventricular Trigeminy',
                                                'Accelerated Idioventricular Rhythm'],

        'Pulmonary Disease': 'Pulmonary Disease',
        'P Wave Changes': ['P Wave Changes','P wave inversion'],
        'QRS - Prolonged': ['QRS - Prolonged',
                            'Poor R Wave Progression',
                            'Intraventricular Conduction Delay',
                            'Idioventricular Rhythm'],
        'QT Interval - Prolonged': 'QT Interval - Prolonged',
        'QT Interval - Short': ['QT Interval - Short','Digitalis Effect'],
        'Q Wave Abnormal': ['Q Wave Abnormal','Q Wave Inversion'],
        'Repolarization Abnormality': 'Repolarization Abnormality',
        'Right Atrial Enlargement': ['Right Atrial Enlargement','Right Atrial Hypertrophy'],
        'Right Axis Deviation': ['Right Axis Deviation','Right Superior Axis','Left Posterior Fascicular Block','Northwest Axis'],
        'Right Bundle Branch Block - General':['Right Bundle Branch Block - General',
                                            'Complete Right Bundle Branch Block',
                                            'Incomplete Right Bundle Branch Block',
                                            'Bifascicular Block'],
        'Right Coronary Artery Occlusion': 'Right Coronary Artery Occlusion',
        'Right Superior Axis': 'Right superior axis',
        'Right Ventricular Hypertrophy': 'Right ventricular hypertrophy',
        'R Wave Abnormal': 'R Wave Abnormal',
        'Septal Infract': 'Septal Infract',
        'ST Changes': ['ST Changes - Nonspecific T-wave Abnormality',
                       'ST changes - Nonspecific ST deviation',
                       'ST changes - Nonspecific ST deviation with T-wave change',
                       'ST-T change due to ventricular hypertrophy',
                       'Inferior ST Segment Depression'],
        'ST Changes - Nonspecific ST Deviation': 'ST changes - Nonspecific ST deviation',
        'ST Changes - Nonspecific ST Deviation With T-wave Change': 'ST changes - Nonspecific ST deviation with T-wave change',
        'ST Changes - Nonspecific T-wave Abnormality': 'ST Changes - Nonspecific T-wave Abnormality',
        'ST Depression': ['ST Depression','Left Ventricular Strain','Digitalis Effect'],
        'ST Elevation': ['STEMI - Anteroseptal',
                    'STEMI - Anterior',
                    'STEMI - Inferior or Inferolateral',
                    'STEMI - Lateral',
                    'STEMI - Posterior',
                    'STEMI - Right Ventricular',
                    'ST Elevation',
                    'Infero-posterior STEMI'],
        'STEMI': ['STEMI - Anteroseptal',
                    'STEMI - Anterior',
                    'STEMI - Inferior or Inferolateral',
                    'STEMI - Lateral',
                    'STEMI - Posterior',
                    'STEMI - Right Ventricular',
                    'Myocardial Infarction - Inferior Or Inferolateral',
                    'Myocardial Infarction - Right Ventricular',
                    'Myocardial Infarction - Anterior',
                    'Myocardial Infarction',
                    'Myocardial Infarction - Anteroseptal',
                    'Myocardial Infarction - Posterior',
                    'Myocardial Infarction - Lateral',
                    'Infero-posterior STEMI',
                    'Septal Infract',
                    'Septal Myocardial Infraction'],
        'Sinosatrial Block': 'Sinosatrial block',
        'Sinus Arrhythmia': 'Sinus arrhythmia',
        'Sinus Bradycardia': 'Sinus bradycardia',
        'Sinus Tachycardia': 'Sinus tachycardia',
        'Supraventricular Tachycardia': ['Supraventricular tachycardia',
                                         'Junctional Tachycardia',
                                         'AV Node Reentrant Tachycardia (AVNRT)'],
        'Typical Atrial Flutter': 'Typical Atrial Flutter',
        'T Wave Abnormal': ['ST-T change due to ventricular hypertrophy',
                            'ST changes - Nonspecific T-wave abnormality',
                            'ST changes - Nonspecific ST deviation with T-wave change',                            
                            'T Wave Inversion',
                            'T Wave Abnormal',
                            'ST Changes - Nonspecific With T-wave Change',
                            'ST changes',
                            'ST changes - Nonspecific',
                            'ST Interval Abnormal',
                            'Hyperacute T waves',
                            'Inferior ST Segment Depression',
                            'De Winter T Wave',
                            'Tall T-Waves',
                            'Left Ventricular Strain'],
        'T Wave Inversion': ['T Wave Inversion',
                             'Left Ventricular Strain'],
        'Tall T-Waves': 'Tall T-Waves',
        'TU Fusion': 'TU Fusion',
        'U Wave Abnormal':'U Wave Abnormal',
        'Ventricular Escape Rhythm': 'Ventricular escape complex(es)',
        'Ventricular Pre Excitation': ['Ventricular Pre Excitation',
                                        'Wolff-Parkinson-White'],
        'Ventricular Tachycardia': ['Ventricular Tachycardia',
                                    'Non Sustained Ventricular Tachycardia',
                                    'Polymorphic Ventricular Tachycardia'],
        'Ventricular Trigeminy': 'Ventricular Trigeminy',
        'Wandering Atrial Pacemaker': 'Wandering atrial pacemaker',
        'Wellens Syndrome': 'Wellens Syndrome',
        'Wide-QRS Tachycardia': ['Ventricular Tachycardia',
                                 'Wide Complex Tachycardia'],
        'Wolff-Parkinson-White': 'Wolff-Parkinson-White'
    }


br_categories_lookup_dict = {
    'AV Block - First-degree': '1dAVb',
    'Atrial Fibrillation': 'AF',
    'Complete Left Bundle Branch Block': 'LBBB',
    'Complete Right Bundle Branch Block': 'RBBB',
    'Sinus Bradycardia': 'SB',
    'Sinus Tachycardia': 'ST'
}

physionet_categories_lookup_dict = {
    'AV Block': ['AVB'],
    'AV Block - First-degree': ['IAVB'],
    'AV Block - Second-degree': ['IIAVB', 'IIAVBII', 'MoI'],
    'AV Block - Third-degree (Complete)': ['CHB'],
    'AV Dissociation': ['AVD'],
    'AV Junctional Rhythm': ['AVJR'],
    'AV Node Reentrant Tachycardia (AVNRT)': ['AVNRT', 'AVRT'],
    'Abnormal QRS': ['abQRS'],
    'Accelerated Atrial Escape Rhythm': ['AAR'],
    'Accelerated Idioventricular Rhythm': ['AIVR'],
    'Accelerated Junctional Rhythm': ['AJR'],
    'Anterior Ischemia': ['AnMIs'],
    'Atrial Bigeminy': ['AB'], #'SVB'
    'Atrial Escape Beat': ['AED'],
    'Atrial Fibrillation': ['AF', 'AFAFL', 'CAF', 'PAF', 'RAF'],
    'Atrial Flutter': ['AFL', 'AFAFL'],
    'Atrial Hypertrophy': ['AH'],
    'Atrial Premature Complex(es) - APC APB': ['PAC', 'BPAC'],
    'Atrial Rhythm': ['ARH', 'SAAWR'],
    'Atrial Tachycardia': ['ATach'],
    'Brugada Syndrome': ['BRU'],
    'Clockwise Or Counterclockwise Vectorcardiographic Loop': ['CVCL/CCVCL','CR','CCR'],
    'Clockwise Rotation': ['CR'],
    'Complete Heart Block': ['CHB'],
    'Complete Left Bundle Branch Block': ['CLBBB'],
    'Complete Right Bundle Branch Block': ['CRBBB'],
    'Coronary Heart Disease': ['CHD'],
    'Countercolockwise Rotation': ['CCR'],
    'Diffuse Intraventricular Block': ['DIB'],
    'Early Repolarization': ['ERe'],
    'Electrode Reversal': ['ALR'],
    'FQRS Wave': ['FQRS'],
    'Fusion Beats': ['FB'],
    'Heart Failure': ['HF'],
    'Heart Valve Disorder': ['HVD'],
    'High T-voltage': ['HTV'],
    'Idioventricular Rhythm': ['IR'],
    'Incomplete Left Bundle Branch Block': ['ILBBB'],
    'Incomplete Right Bundle Branch Block': ['IRBBB'],
    'Inferior Ischaemia': ['IIs'],
    'Inferior ST Segment Depression': ['ISTD'],
    'Junctional Escape': ['JE'],
    'Junctional Premature Complex': ['JPC'],
    'Junctional Tachycardia': ['JTach','AVNRT', 'AVRT'],
    'Lateral Ischemia': ['LIs'],
    'Left Anterior Fascicular Block': ['LAnFB'],
    'Left Atrial Enlargement': ['LAE','LAH'],
    'Left Atrial Hypertrophy': ['LAH','LAE'],
    'Left Axis Deviation': ['LAD','LAnFB'],
    'Left Posterior Fascicular Block': ['LPFB'],
    'Left Ventricular High Voltage': ['LVHV'],
    'Left Ventricular Hypertrophy': ['LVH'],
    'Left Ventricular Strain': ['LVS'],
    'Low QRS Voltages': ['LQRSV'],
    'Myocardial Infarction': ['MI', 'OldMI', 'AMI', 'AnMI'],
    'Myocardial Infarction - Anterior': ['AnMI','AMI'],
    'Myocardial Ischemia': ['MIs', 'AMIs', 'CMI'],
    'Nonspecific Intraventricular Conduction Disorder': ['NSIVCB'],
    'P Wave Changes': ['PWC'],
    'PR Interval - Prolonged': ['LPR','IAVB'],
    'PR Interval - Short': ['SPRI','WPW','AVJR'],
    'Pacing': ['PR', 'AP', 'VPP'],
    'Poor R Wave Progression': ['PRWP'],
    'Premature Atrial Contractions': ['PAC', 'BPAC', 'AB', 'SVB'],
    'Premature Ventricular Contractions': ['VEB', 'PVC','VBig','VTrig'],
    'Prolonged P Wave': ['PPW'],
    'Q Wave Abnormal': ['QAb'],
    'QRS - Prolonged' : ['VTach', 'PVT'],
    'QT Interval - Prolonged': ['LQT'],
    'QT Interval - Short': ['SQT'],
    'R Wave Abnormal': ['RAb'],
    'Right Atrial Abnormality': ['RAAb'],
    'Right Atrial High Voltage': ['RAHV'],
    'Right Atrial Hypertrophy': ['RAH'],
    'Right Axis Deviation': ['RAD','LPFB'],
    'Right Bundle Branch Block - General': ['IRBBB','CRBBB'],
    'Right Ventricular Hypertrophy': ['RVH'],
    'ST Changes': ['STC', 'NSSTTA'],
    'ST Changes - Nonspecific ST Deviation With T-wave Change': ['NSSTTA'],
    'ST Depression': ['STD'],
    'ST Elevation': ['STE'],
    'STEMI': ['STE','MI', 'OldMI', 'AMI', 'AnMI','MIs', 'AMIs', 'CMI'],
    'ST Interval Abnormal': ['STIAb'],
    'Sinosatrial Block': ['SAB'],
    'Sinus Arrhythmia': ['SA'],
    'Sinus Bradycardia': ['SB', 'Brady'],
    'Sinus Node Dysfunction': ['SND'],
    'Sinus Tachycardia': ['STach'],
    'Supraventricular Tachycardia': ['SVT', 'PSVT','JTach','AVNRT', 'AVRT'],
    'T Wave Abnormal': ['TAb','TInv'],
    'T Wave Inversion': ['TInv'],
    'U Wave Abnormal': ['UAb'],
    'Ventricular Bigeminy': ['VBig'],
    'Ventricular Escape Beat': ['VEsB', 'VEsR'],
    'Ventricular Fibrillation': ['VF'],
    'Ventricular Flutter': ['VFL'],
    'Ventricular Pre Excitation': ['VPEx'],
    'Ventricular Tachycardia': ['VTach', 'PVT'],
    'Ventricular Trigeminy': ['VTrig'],
    'Wandering Atrial Pacemaker': ['WAP'],
    'Wide-QRS Tachycardia': ['VTach', 'PVT'],
    'Wolff-Parkinson-White': ['WPW']
}

sph_categories_lookup_dict = {
    'AV Block - Complete': ['AV block_ complete (third-degree)'],
    'AV Block - First-degree': ['AV block_ varying conduction'],
    'AV Block - Second-degree': ['Second-degree AV block_ Mobitz type I (Wenckebach)',
                                 'Second-degree AV block_ Mobitz type II',
                                 'AV block_ advanced (high-grade)',
                                 '2:1 AV block'],
    'AV Conduction Ratio - N:D': ['AV conduction ratio N:D'],
    'Atrial Fibrillation': ['Atrial fibrillation'],
    'Atrial Flutter': ['Atrial flutter'],
    'Atrial Premature Complex(es) - APC APB': ['Atrial premature complex(es)',
                                               'Atrial premature complexes_ nonconducted'],
    'Complete Left Bundle Branch Block': ['Left bundle-branch block'],
    'Complete Right Bundle Branch Block': ['Right bundle-branch block'],
    'Early Repolarization': ['Early repolarization'],
    'Incomplete Right Bundle Branch Block': ['Incomplete right bundle-branch block'],
    'Junctional Premature Complex': ['Junctional premature complex(es)'],
    'Left Anterior Fascicular Block': ['Left anterior fascicular block'],
    'Left Atrial Enlargement': ['Left atrial enlargement'],
    'Left Axis Deviation': ['Left-axis deviation'],
    'Left Posterior Fascicular Block': ['Left posterior fascicular block'],
    'Left Ventricular Hypertrophy': ['Left ventricular hypertrophy'],
    'Low QRS Voltages': ['Low voltage'],
    'Myocardial Infarction': ['Anterior MI',
                              'Anteroseptal MI',
                              'Extensive anterior MI',
                              'Inferior MI'],
    'Myocardial Infarction - Anterior': ['Anterior MI', 'Extensive anterior MI'],
    'Myocardial Infarction - Anteroseptal': ['Anteroseptal MI'],
    'Myocardial Infarction - Inferior Or Inferolateral': ['Inferior MI'],
    'Normal Sinus Rhythm': ['Normal ECG'],
    'PR Interval - Prolonged': ['Prolonged PR interval'],
    'PR Interval - Short': ['Short PR interval'],
    'Premature Ventricular Contractions': ['Ventricular premature complex(es)'],
    'QT Interval - Prolonged': ['Prolonged QT interval'],
    'Right Axis Deviation': ['Right-axis deviation'],
    'Right Ventricular Hypertrophy': ['Right ventricular hypertrophy'],
    'ST Changes': ['ST deviation',
                   'ST deviation with T-wave change',
                   'ST-T change due to ventricular hypertrophy'],
    'ST Changes - Nonspecific ST Deviation': ['ST deviation'],
    'ST Changes - Nonspecific ST Deviation With T-wave Change': ['ST deviation with T-wave change'],
    'Sinus Arrhythmia': ['Sinus arrhythmia'],
    'Sinus Bradycardia': ['Sinus bradycardia'],
    'Sinus Tachycardia': ['Sinus tachycardia'],
    'T Wave Abnormal': ['T-wave abnormality'],
    'TU Fusion': ['TU fusion'],
    'Ventricular Pre Excitation': ['Ventricular preexcitation']

}
def build_mimic_pre_dict():
    mimic_df = pd.read_csv('categories.csv')
    mimic_pre_dict = {}
    for cat in mimic_df['name']:
        mimic_pre_dict[cat] = []
    def add_category(category, aliases):
        if category not in mimic_pre_dict:
            return
        if type(aliases) == str: aliases = [aliases] 
        mimic_pre_dict[category].extend(aliases)

    # Merge categories from physionet_categories_lookup_dict
    for category, aliases in physionet_categories_lookup_dict.items():
        add_category(category, aliases)

    # Merge categories from sph_categories_lookup_dict
    for category, aliases in sph_categories_lookup_dict.items():
        add_category(category, aliases)

    # Merge categories from br_categories_lookup_dict
    for category, aliases in br_categories_lookup_dict.items():
        add_category(category, aliases)

    # Merge categories from Mobile_Labeled_categories_lookup_dict
    for category, aliases in Mobile_Labeled_categories_lookup_dict.items():
        add_category(category, aliases)

    # Merge categories from ny_categories_lookup_dict
    for category, aliases in ny_categories_lookup_dict.items():
        add_category(category, aliases)

    for category, aliases in mimic_pre_dict.items():
        mimic_pre_dict[category] = list(set(aliases))
    return mimic_pre_dict

mimic_pre_dict = build_mimic_pre_dict()
mimic_pre_dict['Premature Ventricular Contractions'] = ['PVC', 'PVCS','PVC(s)']

mimic_categories_lookup_dict = {
    'AV Block': ['AVB'],
    'AV Block - First-degree': ['AV Block - First-degree', 'IAVB', ' type I', '1dAVb', 'AV block_ varying conduction', 'AV Block'],
    'AV Block - Second-degree': ['IIAVBII', 'MoI', ' Mobitz type II', 'AV Block - Second-degree', 'AV block_ advanced (high-grade)', '2:1 AV block', 'IIAVB', ' Mobitz type I (Wenckebach)', 'Second-degree AV block_ Mobitz type I (Wenckebach)', 'Second-degree AV block_ Mobitz type II'],
    'AV Block - Third-degree (Complete)': ['CHB', 'AV Block - Third-degree (Complete)', 'Complete Heart Block', ' complete (third-degree)'],
    'AV Dissociation': ['AV Dissociation', 'AVD'],
    'AV Junctional Rhythm': ['Junctional Escape', 'Accelerated Junctional Rhythm', 'AV Node Reentrant Tachycardia (AVNRT)', 'AV Junctional Rhythm', 'AVJR', 'Junctional Tachycardia'],
    'AV Node Reentrant Tachycardia (AVNRT)': ['AVRT', 'AVNRT'],
    'Abnormal QRS': ['abQRS'],
    'Accelerated Atrial Escape Rhythm': ['AAR'],
    'Accelerated Idioventricular Rhythm': ['AIVR', 'Accelerated Idioventricular Rhythm'],
    'Accelerated Junctional Rhythm': ['Accelerated Junctional Rhythm', 'AJR'],
    'Anterior Ischemia': ['AnMIs', 'Anterior Ischemia'],
    'Atrial Bigeminy': ['AB', 'Atrial Bigeminy'],
    'Atrial Escape Beat': ['AED'],
    'Atrial Fibrillation': ['AF', 'Atrial Fibrillation with a Rapid Ventricular Rate', 'CAF', 'RAF', 'PAF', 'Atrial fibrillation', 'AFAFL', 'Atrial Fibrillation with a Slow Ventricular Rate'],
    'Atrial Flutter': ['AFL', 'Typical Atrial Flutter', 'Atypical Atrial Flutter', 'Atrial flutter', 'AFAFL'],
    'Atrial Hypertrophy': ['Atrial Hypertrophy', 'AH'],
    'Atrial Premature Complex(es) - APC APB': ['Atrial premature complexes_ nonconducted', 'BPAC', 'Atrial Bigeminy', 'PAC', 'Atrial premature complex(es)', 'Atrial premature complex(es) - APC APB', 'Ectopic atrial rhythm', 'Ectopic atrial tachycardia'],
    'Atrial Rhythm': ['ARH', 'SAAWR', 'Accelerated Atrial Escape Rhythm', 'Atrial Rhythm'],
    'Atrial Tachycardia': ['Atrial Tachycardia', 'Ectopic atrial tachycardia', 'ATach', 'Ectopic Atrial Tachycardia'],
    'Brugada Syndrome': ['BRU'],
    'Clockwise Or Counterclockwise Vectorcardiographic Loop': ['CVCL/CCVCL', 'Clockwise Rotation', 'CR', 'Counterclockwise Rotation', 'CCR', 'Clockwise or counterclockwise vectorcardiographic loop'],
    'Clockwise Rotation': ['CR', 'Clockwise Rotation'],
    'Complete Heart Block': ['CHB', 'AV Block - Third-degree (Complete)', 'Complete Heart Block', 'complete (third-degree)'],
    'Complete Left Bundle Branch Block': ['Complete Left Bundle Branch Block', 'CLBBB', 'Left bundle-branch block', 'Bundle Branch Block - Left - LBBB', 'LBBB', 'Left Bundle Branch Block - General'],
    'Complete Right Bundle Branch Block': ['Bundle Branch Block - Right - RBBB', 'Complete Right Bundle Branch Block', 'CRBBB', 'Right Bundle Branch Block - General', 'Right bundle-branch block', 'RBBB'],
    'Coronary Heart Disease': ['CHD'],
    'Countercolockwise Rotation': ['CCR'],
    'Diffuse Intraventricular Block': ['DIB', 'Diffuse Intraventricular Block'],
    'Early Repolarization': ['Early repolarization', 'ERe'],
    'Electrode Reversal': ['Extremity electrode reversal', 'Lead Displacement', 'Electrode Reversal', 'ALR'],
    'FQRS Wave': ['FQRS'],
    'Fusion Beats': ['FB', 'Fusion complex(es)', 'Fusion Beats'],
    'Heart Failure': ['HF'],
    'Heart Valve Disorder': ['HVD'],
    'High T-voltage': ['HTV'],
    'Idioventricular Rhythm': ['Idioventricular Rhythm', 'IR'],
    'Incomplete Left Bundle Branch Block': ['Incomplete Left Bundle Branch Block', 'ILBBB', 'Intermittent Left Bundle Branch Block'],
    'Incomplete Right Bundle Branch Block': ['Incomplete right bundle-branch block', 'IRBBB', 'Incomplete Right Bundle Branch Block', 'Bundle-branch block - RBBB - incomplete'],
    'Inferior Ischaemia': ['Inferior Ischaemia', 'IIs'],
    'Inferior ST Segment Depression': ['Inferior ST Segment Depression', 'ISTD'],
    'Junctional Escape': ['Junctional Escape', 'Junctional escape complex(es)', 'JE'],
    'Junctional Premature Complex': ['Junctional premature complex(es)', 'JPC'],
    'Junctional Tachycardia': ['AV Node Reentrant Tachycardia (AVNRT)', 'AVNRT', 'Junctional tachycardia', 'AVRT', 'JTach', 'Junctional Tachycardia'],
    'Lateral Ischemia': ['Lateral Ischemia', 'LIs'],
    'Left Anterior Fascicular Block': ['Left anterior fascicular block', 'LAnFB'],
    'Left Atrial Enlargement': ['Left Atrial Enlargement', 'LAE', 'LAH', 'Left atrial enlargement', 'Left Atrial Hypertrophy'],
    'Left Atrial Hypertrophy': ['LAH', 'LAE'],
    'Left Axis Deviation': ['Left axis deviation', 'Left anterior fascicular block', 'LAD', 'LAnFB', 'Left-axis deviation'],
    'Left Posterior Fascicular Block': ['Left posterior fascicular block', 'Bifascicular Block', 'LPFB', 'Left Posterior Fascicular Block'],
    'Left Ventricular High Voltage': ['LVHV'],
    'Left Ventricular Hypertrophy': ['Takotsubo Cardiomyopathy', 'Left ventricular hypertrophy', 'Biventricular hypertrophy', 'Left Ventricular Hypertrophy', 'ST-T change due to ventricular hypertrophy', 'LVH'],
    'Left Ventricular Strain': ['LVS'],
    'Low QRS Voltages': ['Low voltage', 'Low QRS Voltages', 'LQRSV'],
    'Myocardial Infarction': ['Myocardial Infarction - Lateral', 'STEMI - Anterior', 'Inferior MI', 'Infero-posterior STEMI', 'STEMI - Lateral', 'STEMI - Posterior', 'STEMI - Inferior or Inferolateral', 'Myocardial Infarction', 'Myocardial Infarction - Right Ventricular', 'Myocardial Infarction - Anterior', 'Myocardial Infarction - Posterior', 'Anteroseptal MI', 'STEMI - Anteroseptal', 'OldMI', 'Septal Infract', 'Extensive anterior MI', 'STEMI - Right Ventricular', 'AnMI', 'Anterior MI', 'AMI', 'Septal Myocardial Infraction', 'Myocardial Infarction - Anteroseptal', 'Myocardial Infarction - Inferior Or Inferolateral', 'MI'],
    'Myocardial Infarction - Anterior': ['STEMI - Anterior', 'AnMI', 'Anterior MI', 'Extensive anterior MI', 'AMI', 'STEMI - Anteroseptal'],
    'Myocardial Infarction - Inferior Or Inferolateral': ['Inferior MI', 'Myocardial Infarction - Inferior Or Inferolateral', 'STEMI - Inferior or Inferolateral'],
    'Normal Sinus Rhythm': ['Normal ECG'],
    'ST Changes - Nonspecific ST Deviation': ['ST deviation', 'ST changes - Nonspecific ST deviation'],
    'TU Fusion': ['TU Fusion', 'TU fusion'],
    'Aberrant Conduction of Supraventricular Beat(s)': ['Aberrant Conduction of Supraventricular Beat(s)'],
    'Abnormal P-wave Axis': ['Abnormal P-wave axis'],
    'Atrial Pacing': ['Atrial Pacing'],
    'Acute Myocardial Ischemia': ['Acute Myocardial Ischemia'],
    'Acute Pericarditis': ['Acute pericarditis', 'Acute Pericarditis'],
    'Anterolateral Ischemia': ['Anterolateral Ischemia'],
    'Atrial Fibrillation & Flutter': ['Typical Atrial Flutter', 'Atypical Atrial Flutter', 'Atrial Fibrillation with a Rapid Ventricular Rate', 'Atrial flutter', 'Atrial fibrillation', 'Atrial Fibrillation with a Slow Ventricular Rate'],
    'Bifascicular Block': ['Bifascicular block'],
    'Counterclockwise Rotation': ['Counterclockwise Rotation'],
    'Dextrocardia': ['Dextrocardia'],
    'De Winter T Wave': ['De Winter T Wave'],
    'Digitalis Effect': ['Digitalis effect', 'Digitalis Effect'],
    'Early Q wave Formation': ['Early Q wave formation'],
    'Ectopic Atrial Tachycardia': ['Ectopic Atrial Tachycardia', 'Ectopic atrial tachycardia'],
    'Escape Bigeminy': ['Escape Bigeminy'],
    'Hyperacute T waves': ['Hyperacute T waves'],
    'Infero-posterior STEMI': ['Infero-posterior STEMI'],
    'Intraventricular Conduction Delay': ['Intraventricular Conduction Delay', 'QRS - Prolonged', 'Wide-QRS tachycardia', 'Poor R Wave Progression', 'Idioventricular Rhythm', 'Intraventricular conduction delay', 'Wide Complex Tachycardia'],
    'Ischemia': ['Lateral Ischemia', 'Acute Myocardial Ischemia', 'Ischemia - location unspecified', 'Myocardial Ischemia', 'Inferior Ischaemia', 'Anterolateral Ischemia', 'Anterior IschemiaInferior Ischemia', 'Ischemia'],
    'J Wave (Osborn Wave)': ['J Wave (Osborn Wave)'],
    'Left Bundle Branch Block': ['Complete Left Bundle Branch Block', 'Intermittent Left Bundle Branch Block', 'Incomplete Left Bundle Branch Block', 'Bundle Branch Block - Left - LBBB', 'Left Bundle Branch Block - General'],
    'Myocardial Infarction - Lateral': ['Myocardial Infarction - Lateral', 'STEMI - Lateral'],
    'Myocardial Infarction - Posterior': ['STEMI - Posterior', 'Myocardial Infarction - Posterior'],
    'Myocardial Infarction - Right Ventricular': ['STEMI - Right Ventricular', 'Myocardial Infarction - Right Ventricular'],
    'Myocardial Infarction - Septal': ['Septal Infract', 'STEMI - Anteroseptal', 'Septal Myocardial Infraction', 'Myocardial Infarction - Anteroseptal'],
    'Normal ECG': ['Normal Sinus Rhythm'],
    'Pericarditis': ['Acute Pericarditis', 'Pericarditis'],
    'Posterior Ischemia': ['Posterior Ischemia'],
    'Right Axis Deviation': ['Right Axis Deviation', 'RAD'],
    'Right Bundle Branch Block - General': ['Bundle Branch Block - Right - RBBB', 'Complete Right Bundle Branch Block', 'Incomplete Right Bundle Branch Block', 'Right bundle-branch block', 'CRBBB', 'Right bundle-branch block - incomplete', 'Right Bundle Branch Block - General'],
    'Right Ventricular Hypertrophy': ['Right Ventricular Hypertrophy', 'RVH'],
    'Right Ventricular Strain': ['Right Ventricular Strain', 'RVS'],
    'ST Segment Depression': ['ST segment depression'],
    'ST Segment Elevation': ['ST segment elevation'],
    'ST-elevation Myocardial Infarction (STEMI)': ['STEMI - Anterior', 'Inferior MI', 'Infero-posterior STEMI', 'STEMI - Lateral', 'STEMI - Posterior', 'STEMI - Inferior or Inferolateral', 'STEMI - Right Ventricular', 'Myocardial Infarction - Lateral', 'Myocardial Infarction - Posterior', 'STEMI', 'STEMI - Anteroseptal'],
    'T Wave Abnormal': ['Hyperacute T waves', 'T-wave inversion', 'TWA', 'T Wave Abnormal', 'TWI'],
    'T Wave Alternans': ['T Wave Alternans'],
    'Takotsubo Cardiomyopathy': ['Takotsubo Cardiomyopathy', 'Left ventricular hypertrophy', 'Biventricular hypertrophy', 'Left Ventricular Hypertrophy', 'ST-T change due to ventricular hypertrophy', 'LVH'],
    'Unconfirmed Myocardial Infarction': ['OldMI', 'Unconfirmed Myocardial Infarction'],
    'Ventricular Bigeminy': ['Ventricular bigeminy', 'VB'],
    'Ventricular Escape Beat': ['Ventricular escape complex(es)', 'VEB'],
    'Ventricular Escape Rhythm': ['Ventricular Escape Rhythm', 'VER'],
    'Ventricular Paced Rhythm': ['Ventricular Paced Rhythm', 'VPR'],
    'Ventricular Tachycardia': ['Ventricular Tachycardia', 'Vtach'],
    'Wide Complex Tachycardia': ['Wide Complex Tachycardia', 'Intraventricular Conduction Delay', 'QRS - Prolonged', 'Poor R Wave Progression', 'Idioventricular Rhythm', 'Intraventricular conduction delay', 'Wide-QRS tachycardia']
}

dataset_lookup_dicts = {
    'NY': Categories(ny_categories_lookup_dict),
    'Brazilian': Categories(br_categories_lookup_dict),
    'SPH': Categories(sph_categories_lookup_dict),
    'CPSC': Categories(physionet_categories_lookup_dict),
    'CPSC_Extra': Categories(physionet_categories_lookup_dict),
    'StPetersburg': Categories(physionet_categories_lookup_dict),
    'PTB': Categories(physionet_categories_lookup_dict),
    'PTB_XL': Categories(physionet_categories_lookup_dict),
    'Georgia': Categories(physionet_categories_lookup_dict),
    'Chapman_Shaoxing': Categories(physionet_categories_lookup_dict),
    'Ningbo': Categories(physionet_categories_lookup_dict),
    'Mobile_Labeled': Categories(Mobile_Labeled_categories_lookup_dict)
}
