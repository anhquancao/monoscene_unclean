import torch
import numpy as np

NYU_class_names = ['empty','ceiling','floor','wall','window','chair','bed','sofa','table','tvs','furn','objs']
#class_weights = torch.FloatTensor([0.01, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
class_weights = torch.FloatTensor([0.05, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
#class_relation_weights = torch.ones(78).float()
#class_relation_weights[:12] = 0.1
#class_relation_weights = torch.FloatTensor([1, 0.03, 0.1, 0.1])
#class_relation_weights = torch.FloatTensor([1.0, 0.1, 1.0, 1.0, 1.0])
#class_relation_weights = torch.FloatTensor([5.0, 0.1, 0.3, 3.0, 0.3])
#class_relation_weights = torch.FloatTensor([10.0, 0.5, 1.0, 4.0, 1.0])
class_relation_weights = torch.FloatTensor([0.66, 0.29, 0.02, 0.03])
invfreq_class_relation_weights = torch.FloatTensor([0.59495053, 0.01930607, 0.06316538, 0.25941265, 0.06316538]) # 1/freq
#invfreq_class_relation_weights = torch.FloatTensor([1.0, 1.5]) # freq: 0.6, 0.4
#class_relation_weights = torch.FloatTensor([0.06829726231, 0.05669879682, 0.05955394416, 0.06023715249])

class_freq_1_4 = np.array([43744234, 80205, 1070052, 905632, 116952, 180994, 436852, 279714, 254611, 28247, 1805949, 850724])
class_freq_1_8 = np.array([5176253, 17277, 220105, 183849, 21827, 33520, 67022, 44248, 46615, 4419,  290218,  142573])
class_freq_1_16 = np.array([587620, 3820, 46836, 36256, 4241, 5978, 10939, 8000, 8224, 781, 49778,  25864])

class_relation_freqs = np.array([936560098 ,326268 ,724848 ,371900 ,54040 ,119848 ,19448 ,56372 ,179278 ,764 ,458400 ,383946 ,4381514 ,4683142 ,490996 ,1121002 ,838800 ,1182996 ,1449626 ,98522 ,5969538 ,3573824 ,2544868 ,410488 ,618514 ,1353244 ,841166 ,794614 ,100116 ,4203680 ,2317706 ,187185 ,69628 ,177598 ,114936 ,80874 ,16278 ,392814 ,215816 ,193194 ,29376 ,96236 ,312542 ,3122 ,637158 ,375024 ,1005735 ,46698 ,56686 ,11252 ,922274 ,574582 ,641250 ,242914 ,25712 ,664396 ,475882 ,551720 ,4314 ,810524 ,530230 ,17223 ,118644 ,42268 ,7232478 ,3788946 ,1797694])

NYU_class_cluster_4 = {
            # 'empty',
            0: 0,
            # 'ceiling',
            1: 1,
            # 'floor',
            2: 1,
            # 'wall',
            3: 1,
            # 'window',
            4: 1,
            # 'chair',
            5: 2,
            # 'bed',
            6: 2,
            # 'sofa',
            7: 2,
            # 'table',
            8: 2,
            # 'tvs',
            9: 3,
            # 'furn',
            10: 2,
            # 'objs'            
            11: 3
        }

# NYU_class_cluster_4 = {
#             # 'empty',
#             0: 0,
#             # 'ceiling',
#             1: 1,
#             # 'floor',
#             2: 1,
#             # 'wall',
#             3: 1,
#             # 'window',
#             4: 1,
#             # 'chair',
#             5: 2,
#             # 'bed',
#             6: 2,
#             # 'sofa',
#             7: 2,
#             # 'table',
#             8: 2,
#             # 'tvs',
#             9: 3,
#             # 'furn',
#             10: 3,
#             # 'objs'            
#             11: 3
#         }



NYU_class_cluster_6 = {
            # 'empty',
            0: 0,
            # 'ceiling',
            1: 1,
            # 'floor',
            2: 1,
            # 'wall',
            3: 2,
            # 'window',
            4: 2,
            # 'chair',
            5: 4,
            # 'bed',
            6: 3,
            # 'sofa',
            7: 3,
            # 'table',
            8: 4,
            # 'tvs',
            9: 5,
            # 'furn',
            10: 4,
            # 'objs'            
            11: 5
        }