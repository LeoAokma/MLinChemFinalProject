# Declaring features:
test_key = ['time', 'slowCool', 'pH', 'leak', 'numberInorg', 'numberOrg',
            'numberOxlike', 'numberComponents', 'orgavgpolMax',
            'orgrefractivityMax', 'orgmaximalprojectionareaMax',
            'orgmaximalprojectionradiusMax', 'orgmaximalprojectionsizeMax',
            'orgminimalprojectionareaMax', 'orgminimalprojectionradiusMax',
            'orgminimalprojectionsizeMax', 'orgavgpol_pHdependentMax',
            'orgmolpolMax', 'orgvanderwaalsMax', 'orgASAMax', 'orgASA+Max',
            'orgASA-Max', 'orgASA_HMax', 'orgASA_PMax', 'orgpolarsurfaceareaMax',
            'orghbdamsaccMax', 'orghbdamsdonMax', 'orgavgpolMin', 'orgrefractivityMin',
            'orgmaximalprojectionareaMin', 'orgmaximalprojectionradiusMin',
            'orgmaximalprojectionsizeMin', 'orgminimalprojectionareaMin',
            ]
feat_first = ['orgvanderwaalsMin', 'orgASA+Min', 'orghbdamsdonGeomAvg',
              'PaulingElectronegMean', 'hardnessMaxWeighted', 'AtomicRadiusMeanWeighted']

feat_top6 = ['time', 'hardnessMinWeighted', 'orgASA_HGeomAvg', 'leak',
             'inorg-water-moleratio', 'orgvanderwaalsArithAvg']

feat_top9 = ['time', 'hardnessMinWeighted', 'orgASA_HGeomAvg', 'leak',
             'inorg-water-moleratio', 'orgvanderwaalsArithAvg', 'orgvanderwaalsArithAvg',
             'orghbdamsaccMax', 'temp', 'EAMinWeighted']
