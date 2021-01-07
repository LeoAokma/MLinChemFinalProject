# Declaring features:
test_key = [['time', 'slowCool', 'pH', 'leak', 'numberInorg', 'numberOrg',
            'numberOxlike', 'numberComponents', 'orgavgpolMax',
            'orgrefractivityMax', 'orgmaximalprojectionareaMax',
            'orgmaximalprojectionradiusMax', 'orgmaximalprojectionsizeMax',
            'orgminimalprojectionareaMax', 'orgminimalprojectionradiusMax',
            'orgminimalprojectionsizeMax', 'orgavgpol_pHdependentMax',
            'orgmolpolMax', 'orgvanderwaalsMax', 'orgASAMax', 'orgASA+Max',
            'orgASA-Max', 'orgASA_HMax', 'orgASA_PMax', 'orgpolarsurfaceareaMax',
            'orghbdamsaccMax', 'orghbdamsdonMax', 'orgavgpolMin', 'orgrefractivityMin',
            'orgmaximalprojectionareaMin', 'orgmaximalprojectionradiusMin',
            'orgmaximalprojectionsizeMin', 'orgminimalprojectionareaMin'],
            ['反应时间', '慢速冷却', 'pH', 'leak', '无机组分数', '有机组分数量',
             'Oxlike数', '组分数', 'orgavgpolMax',
            'orgrefractivityMax', 'orgmaximalprojectionareaMax',
            'orgmaximalprojectionradiusMax', 'orgmaximalprojectionsizeMax',
            'orgminimalprojectionareaMax', 'orgminimalprojectionradiusMax',
            'orgminimalprojectionsizeMax', 'orgavgpol_pHdependentMax',
            'orgmolpolMax', 'orgvanderwaalsMax', 'orgASAMax', 'orgASA+Max',
            'orgASA-Max', 'orgASA_HMax', 'orgASA_PMax', 'orgpolarsurfaceareaMax',
            'orghbdamsaccMax', 'orghbdamsdonMax', 'orgavgpolMin', 'orgrefractivityMin',
            'orgmaximalprojectionareaMin', 'orgmaximalprojectionradiusMin',
            'orgmaximalprojectionsizeMin', 'orgminimalprojectionareaMin'
             ]
            ]
feat_first = [['orgvanderwaalsMin', 'orgASA+Min', 'orghbdamsdonGeomAvg',
              'PaulingElectronegMean', 'hardnessMaxWeighted', 'AtomicRadiusMeanWeighted'],
              ['orgMin', 'orgASA+Min', 'orghbdamsdonGeom平均值',
               '平均Pauling电负性', '硬度MaxWeighted', '原子半径MeanWeighted']
              ]

feat_top6 = [['time', 'hardnessMinWeighted', 'orgASA_HGeomAvg', 'leak',
             'inorg-water-moleratio', 'orgvanderwaalsArithAvg'],
             ['反应时间', '硬度MinWeighted', 'orgASA_H几何平均', 'leak',
             '无机组分/水摩尔比', 'org范德华算术平均']
             ]

feat_top9 = [['time', 'hardnessMinWeighted', 'orgASA_HGeomAvg', 'leak',
             'inorg-water-moleratio', 'orgvanderwaalsArithAvg',
             'orghbdamsaccMax', 'temp', 'EAMinWeighted'],
             ['反应时间', '硬度MinWeighted', 'orgASA_H几何平均', 'leak',
             '无机组分/水摩尔比', 'org范德华算术平均值',
             'orghbdamsacc最大值', '反应温度', 'EAMinWeighted']
             ]
