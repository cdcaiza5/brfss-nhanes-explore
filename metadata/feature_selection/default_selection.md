# Seleccion del preset `default` — 2026-07-20

Seleccion de las 30 mejores variables para `ADDEPEV3` sobre el universo
completo de columnas (menos ID/geo/peso/estructurales y el target).
Generado por `tools/select_rf_top30.py` (seed 42, subsample 100,000).

## Techo de ranking

- RF sobre el universo completo (train): **0.9039**
- RF sobre el universo completo (test): **0.8407**
- RF sobre el top-30 canonico (test): **0.8331**
  (delta vs universo: -0.0077)

## RandomForest — ranking completo (canonico: top-30)

| rank | variable | importancia RF |
|---|---|---|
| 1 | `MENTHLTH` | 0.15857 |
| 2 | `_MENT14D` | 0.14782 |
| 3 | `POORHLTH` | 0.07117 |
| 4 | `DECIDE` | 0.05926 |
| 5 | `_PHYS14D` | 0.02364 |
| 6 | `PHYSHLTH` | 0.02300 |
| 7 | `GENHLTH` | 0.01447 |
| 8 | `SDLONELY` | 0.01446 |
| 9 | `DIFFALON` | 0.01355 |
| 10 | `_RFHLTH` | 0.01305 |
| 11 | `_SEX` | 0.01217 |
| 12 | `SEXVAR` | 0.01164 |
| 13 | `ECIGNOW3` | 0.01133 |
| 14 | `_AGE80` | 0.01007 |
| 15 | `EMPLOY1` | 0.00950 |
| 16 | `ASTHMA3` | 0.00900 |
| 17 | `_ASTHMS1` | 0.00887 |
| 18 | `_AGEG5YR` | 0.00877 |
| 19 | `_BMI5` | 0.00876 |
| 20 | `LSATISFY` | 0.00815 |
| 21 | `HIVTST7` | 0.00791 |
| 22 | `_LTASTH1` | 0.00717 |
| 23 | `_DRDXAR2` | 0.00714 |
| 24 | `CELLSEX3` | 0.00712 |
| 25 | `_AIDTST4` | 0.00676 |
| 26 | `HIVTSTD3` | 0.00667 |
| 27 | `HEIGHT3` | 0.00649 |
| 28 | `HAVARTH4` | 0.00638 |
| 29 | `WTKG3` | 0.00637 |
| 30 | `_STSTR` | 0.00632 |
| 31 | `_RACE_enc` | 0.00605 |
| 32 | `_CASTHM1` | 0.00572 |
| 33 | `_AGE_G` | 0.00541 |
| 34 | `HTM4` | 0.00509 |
| 35 | `HTIN4` | 0.00506 |
| 36 | `_PACKYRS` | 0.00502 |
| 37 | `INCOME3` | 0.00486 |
| 38 | `_HCVU654` | 0.00485 |
| 39 | `_LCSYSMK` | 0.00448 |
| 40 | `MARITAL` | 0.00442 |
| 41 | `IDAY` | 0.00441 |
| 42 | `_LCSAGE` | 0.00434 |
| 43 | `_AGE65YR` | 0.00431 |
| 44 | `LCSFIRST` | 0.00418 |
| 45 | `RENTHOM1` | 0.00417 |
| 46 | `DIFFWALK` | 0.00407 |
| 47 | `MEDCOST1` | 0.00380 |
| 48 | `PRIMINS2` | 0.00374 |
| 49 | `_LCSSMKG` | 0.00359 |
| 50 | `_RACEGR3` | 0.00348 |
| 51 | `CRVCLCNC` | 0.00345 |
| 52 | `_SMOKER3` | 0.00341 |
| 53 | `_DUALCOR` | 0.00336 |
| 54 | `_RACEPRV` | 0.00334 |
| 55 | `EMTSUPRT` | 0.00327 |
| 56 | `CHCCOPD3` | 0.00327 |
| 57 | `IMONTH` | 0.00320 |
| 58 | `_IMPRACE` | 0.00310 |
| 59 | `FMONTH` | 0.00308 |
| 60 | `_INCOMG1` | 0.00301 |
| 61 | `_DRNKWK3` | 0.00299 |
| 62 | `_RACEG21` | 0.00294 |
| 63 | `LCSLAST_` | 0.00289 |
| 64 | `SMOKE100` | 0.00288 |
| 65 | `_LCSYQTS` | 0.00287 |
| 66 | `QSTVER` | 0.00274 |
| 67 | `_MRACE1` | 0.00272 |
| 68 | `PERSDOC3` | 0.00266 |
| 69 | `DROCDY4_` | 0.00261 |
| 70 | `FLSHTMY3` | 0.00261 |
| 71 | `LCSNUMCG` | 0.00260 |
| 72 | `ALCDAY4` | 0.00254 |
| 73 | `_BMI5CAT` | 0.00250 |
| 74 | `DIABAGE4` | 0.00241 |
| 75 | `_PACKDAY` | 0.00237 |
| 76 | `LCSNUMC_` | 0.00225 |
| 77 | `MAXDRNKS` | 0.00221 |
| 78 | `HHADULT` | 0.00208 |
| 79 | `EDUCA` | 0.00206 |
| 80 | `CRVCLHPV` | 0.00193 |
| 81 | `_EDUCAG` | 0.00184 |
| 82 | `COLNTES1` | 0.00180 |
| 83 | `_DUALUSE` | 0.00177 |
| 84 | `AVEDRNK4` | 0.00176 |
| 85 | `_PNEUMO3` | 0.00175 |
| 86 | `RMVTETH4` | 0.00173 |
| 87 | `_FLSHOT7` | 0.00171 |
| 88 | `FLUSHOT7` | 0.00170 |
| 89 | `IMFVPLA5` | 0.00167 |
| 90 | `CHECKUP1` | 0.00164 |
| 91 | `PNEUVAC4` | 0.00155 |
| 92 | `_LCSCTSN` | 0.00152 |
| 93 | `LASTDEN4` | 0.00147 |
| 94 | `_CHLDCNT` | 0.00146 |
| 95 | `QSTLANG` | 0.00143 |
| 96 | `LCSCTSC1` | 0.00141 |
| 97 | `SDHFOOD1` | 0.00140 |
| 98 | `_CRVSCRN` | 0.00140 |
| 99 | `SSBSUGR2` | 0.00139 |
| 100 | `SOFEMALE` | 0.00138 |
| 101 | `HOWLONG` | 0.00138 |
| 102 | `SSBFRUT3` | 0.00134 |
| 103 | `CIMEMLO1` | 0.00133 |
| 104 | `MARIJAN1` | 0.00127 |
| 105 | `CHILDREN` | 0.00125 |
| 106 | `HADMAM` | 0.00123 |
| 107 | `_RFBMI5` | 0.00120 |
| 108 | `HOWSAFE1` | 0.00119 |
| 109 | `HADHYST2` | 0.00118 |
| 110 | `DRNK3GE5` | 0.00114 |
| 111 | `PDIABTS1` | 0.00111 |
| 112 | `DIFFDRES` | 0.00110 |
| 113 | `CPDEMO1C` | 0.00107 |
| 114 | `EXERANY2` | 0.00105 |
| 115 | `HIVRISK5` | 0.00104 |
| 116 | `HADSIGM4` | 0.00102 |
| 117 | `DIABETE4` | 0.00102 |
| 118 | `MSCODE` | 0.00100 |
| 119 | `_EXTETH3` | 0.00100 |
| 120 | `_RAWRAKE` | 0.00098 |
| 121 | `_CURECI3` | 0.00097 |
| 122 | `_DENVST3` | 0.00095 |
| 123 | `_TOTINDA` | 0.00093 |
| 124 | `SMOKDAY2` | 0.00093 |
| 125 | `_CRCREC3` | 0.00091 |
| 126 | `_HADCOLN` | 0.00090 |
| 127 | `FOODSTMP` | 0.00090 |
| 128 | `_HISPANC` | 0.00087 |
| 129 | `LCSSCNCR` | 0.00087 |
| 130 | `STOLTEST` | 0.00087 |
| 131 | `BLIND` | 0.00086 |
| 132 | `NUMADULT` | 0.00083 |
| 133 | `_CLNSCP2` | 0.00082 |
| 134 | `SDHEMPLY` | 0.00080 |
| 135 | `DEAF` | 0.00078 |
| 136 | `LASTSMK2` | 0.00077 |
| 137 | `LCSCTWHN` | 0.00077 |
| 138 | `COLNCNCR` | 0.00076 |
| 139 | `_RFBING6` | 0.00076 |
| 140 | `LANDLINE` | 0.00075 |
| 141 | `SIGMTES1` | 0.00075 |
| 142 | `_RFBLDS6` | 0.00074 |
| 143 | `_RFDRHV9` | 0.00070 |
| 144 | `_SBONTI2` | 0.00070 |
| 145 | `_METSTAT` | 0.00069 |
| 146 | `CHCOCNC1` | 0.00068 |
| 147 | `_CHISPNC` | 0.00067 |
| 148 | `SOMALE` | 0.00066 |
| 149 | `SDHTRNSP` | 0.00066 |
| 150 | `DRNKANY6` | 0.00066 |
| 151 | `LANDSEX3` | 0.00066 |
| 152 | `CRGVPRB4` | 0.00065 |
| 153 | `CAGEG` | 0.00064 |
| 154 | `CHCKDNY2` | 0.00064 |
| 155 | `VETERAN3` | 0.00063 |
| 156 | `_RFSMOK3` | 0.00063 |
| 157 | `COLNSIGM` | 0.00063 |
| 158 | `FIREARM5` | 0.00062 |
| 159 | `_MICHD` | 0.00062 |
| 160 | `CNCRAGE` | 0.00062 |
| 161 | `_CRACE1` | 0.00061 |
| 162 | `CERVSCRN` | 0.00060 |
| 163 | `ASTHNOW` | 0.00060 |
| 164 | `_RFMAM23` | 0.00059 |
| 165 | `_ALTETH3` | 0.00058 |
| 166 | `SDHUTILS` | 0.00058 |
| 167 | `CVDCRHD4` | 0.00058 |
| 168 | `SDHBILLS` | 0.00058 |
| 169 | `DISPCODE` | 0.00056 |
| 170 | `PREDIAB2` | 0.00054 |
| 171 | `CRGVREL5` | 0.00054 |
| 172 | `CHCSCNC1` | 0.00053 |
| 173 | `_URBSTAT` | 0.00052 |
| 174 | `USENOW3` | 0.00051 |
| 175 | `TYPCNTR9` | 0.00051 |
| 176 | `CNCRTYP2` | 0.00051 |
| 177 | `_HPV5YR1` | 0.00051 |
| 178 | `_SGMSCP2` | 0.00050 |
| 179 | `_LCSPSTF` | 0.00050 |
| 180 | `ACESWEAR` | 0.00049 |
| 181 | `CHKHEMO3` | 0.00049 |
| 182 | `CVDSTRK3` | 0.00049 |
| 183 | `MARJEAT` | 0.00048 |
| 184 | `CDHOUS1` | 0.00047 |
| 185 | `CVDINFR4` | 0.00046 |
| 186 | `RCSRLTN2` | 0.00046 |
| 187 | `_SGMS102` | 0.00046 |
| 188 | `HEATTBCO` | 0.00045 |
| 189 | `CRGVLNG2` | 0.00045 |
| 190 | `_HADSIGM` | 0.00044 |
| 191 | `ACEDEPRS` | 0.00044 |
| 192 | `ACEADSAF` | 0.00043 |
| 193 | `ACETOUCH` | 0.00040 |
| 194 | `DIABEYE1` | 0.00040 |
| 195 | `_MAM402Y` | 0.00040 |
| 196 | `MARJVAPE` | 0.00039 |
| 197 | `VCLNTES2` | 0.00039 |
| 198 | `VIRCOLO1` | 0.00039 |
| 199 | `_HLTHPL2` | 0.00038 |
| 200 | `EYEEXAM1` | 0.00038 |
| 201 | `CRVCLPAP` | 0.00038 |
| 202 | `ACEHURT1` | 0.00038 |
| 203 | `CAREGIV1` | 0.00037 |
| 204 | `DIABEDU1` | 0.00036 |
| 205 | `SMALSTOL` | 0.00036 |
| 206 | `SDNATES1` | 0.00036 |
| 207 | `GUNLOAD` | 0.00035 |
| 208 | `STOOLDN2` | 0.00035 |
| 209 | `IYEAR` | 0.00035 |
| 210 | `CSTATE1` | 0.00035 |
| 211 | `_RFPAP37` | 0.00034 |
| 212 | `_STOLDN2` | 0.00034 |
| 213 | `CDSOCIA1` | 0.00034 |
| 214 | `ACEPUNCH` | 0.00033 |
| 215 | `TETANUS1` | 0.00032 |
| 216 | `ACEDIVRC` | 0.00032 |
| 217 | `USEMRJN4` | 0.00031 |
| 218 | `ACEADNED` | 0.00030 |
| 219 | `ACEDRINK` | 0.00030 |
| 220 | `_LCSELIG` | 0.00030 |
| 221 | `PCSTALK2` | 0.00030 |
| 222 | `CRGVHRS2` | 0.00029 |
| 223 | `ARTHEXER` | 0.00029 |
| 224 | `MARJSMOK` | 0.00028 |
| 225 | `CDDISCU1` | 0.00028 |
| 226 | `CASTHDX2` | 0.00028 |
| 227 | `_PAPHPV1` | 0.00028 |
| 228 | `CDWORRY` | 0.00026 |
| 229 | `PSATEST1` | 0.00026 |
| 230 | `STOPSMK2` | 0.00026 |
| 231 | `MENTCIGS` | 0.00025 |
| 232 | `ACETTHEM` | 0.00025 |
| 233 | `BLDSTFIT` | 0.00022 |
| 234 | `_VIRCOL2` | 0.00022 |
| 235 | `CRGVPER2` | 0.00021 |
| 236 | `ACEDRUGS` | 0.00019 |
| 237 | `ACEHVSEX` | 0.00018 |
| 238 | `PSATIME1` | 0.00018 |
| 239 | `LOADULK2` | 0.00018 |
| 240 | `NUMHHOL4` | 0.00018 |
| 241 | `HPVADVC4` | 0.00017 |
| 242 | `CRGVHOU2` | 0.00017 |
| 243 | `LASTSIG4` | 0.00017 |
| 244 | `HADSEX` | 0.00017 |
| 245 | `SHINGLE2` | 0.00017 |
| 246 | `ACEPRISN` | 0.00016 |
| 247 | `INSULIN1` | 0.00016 |
| 248 | `DIABTYPE` | 0.00016 |
| 249 | `NOBCUSE8` | 0.00016 |
| 250 | `MENTECIG` | 0.00015 |
| 251 | `CRGVNURS` | 0.00014 |
| 252 | `NUMPHON4` | 0.00014 |
| 253 | `CRGVALZD` | 0.00013 |
| 254 | `CSRVDOC1` | 0.00013 |
| 255 | `CSRVTRT3` | 0.00013 |
| 256 | `CNCRDIFF` | 0.00013 |
| 257 | `FEETSORE` | 0.00012 |
| 258 | `HPVADSH1` | 0.00012 |
| 259 | `PFPPRVN4` | 0.00010 |
| 260 | `PCPSARS2` | 0.00010 |
| 261 | `HPVDSHT` | 0.00010 |
| 262 | `PREGNANT` | 0.00009 |
| 263 | `MARJOTHR` | 0.00009 |
| 264 | `MARJDAB` | 0.00009 |
| 265 | `CASTHNO2` | 0.00008 |
| 266 | `RESPSLC1` | 0.00008 |
| 267 | `CSRVRTRN` | 0.00008 |
| 268 | `PVTRESD3` | 0.00008 |
| 269 | `CSRVSUM` | 0.00007 |
| 270 | `PSASUGS1` | 0.00006 |
| 271 | `CSRVPAIN` | 0.00005 |
| 272 | `CSRVDEIN` | 0.00003 |
| 273 | `CSRVCTL2` | 0.00002 |
| 274 | `ICFQSTVR` | 0.00002 |
| 275 | `CSRVINSR` | 0.00002 |
| 276 | `CSRVINST` | 0.00002 |
| 277 | `CSRVCLIN` | 0.00001 |
| 278 | `LADULT1` | 0.00001 |
| 279 | `PVTRESD1` | 0.00000 |
| 280 | `CCLGHOUS` | 0.00000 |
| 281 | `COLGHOUS` | 0.00000 |
| 282 | `CTELENM1` | 0.00000 |
| 283 | `CADULT1` | 0.00000 |
| 284 | `CELLFON5` | 0.00000 |
| 285 | `STATERE1` | 0.00000 |
| 286 | `CELPHON1` | 0.00000 |
| 287 | `CTELNUM1` | 0.00000 |
| 288 | `SAFETIME` | 0.00000 |

## Mutual Information — ranking completo (contraste; canonico: top-30)

| rank | variable | MI |
|---|---|---|
| 1 | `_BMI5` | 0.14901 |
| 2 | `_STSTR` | 0.14204 |
| 3 | `MENTHLTH` | 0.10023 |
| 4 | `_MENT14D` | 0.09472 |
| 5 | `POORHLTH` | 0.04895 |
| 6 | `DECIDE` | 0.04857 |
| 7 | `HIVTSTD3` | 0.04708 |
| 8 | `WTKG3` | 0.03341 |
| 9 | `PHYSHLTH` | 0.02618 |
| 10 | `GENHLTH` | 0.02242 |
| 11 | `_PHYS14D` | 0.02221 |
| 12 | `EMPLOY1` | 0.01908 |
| 13 | `_AGE80` | 0.01819 |
| 14 | `DIFFALON` | 0.01818 |
| 15 | `_PACKYRS` | 0.01610 |
| 16 | `_RFHLTH` | 0.01521 |
| 17 | `_ASTHMS1` | 0.01358 |
| 18 | `MEDCOST1` | 0.01357 |
| 19 | `_LTASTH1` | 0.01345 |
| 20 | `ASTHMA3` | 0.01345 |
| 21 | `HIVTST7` | 0.01322 |
| 22 | `_AIDTST4` | 0.01322 |
| 23 | `DIABAGE4` | 0.01293 |
| 24 | `_AGEG5YR` | 0.01232 |
| 25 | `_LCSYQTS` | 0.01219 |
| 26 | `_SEX` | 0.01195 |
| 27 | `SEXVAR` | 0.01192 |
| 28 | `ECIGNOW3` | 0.01189 |
| 29 | `HEIGHT3` | 0.01189 |
| 30 | `_DUALCOR` | 0.01163 |
| 31 | `CRVCLHPV` | 0.01138 |
| 32 | `_CASTHM1` | 0.01102 |
| 33 | `HTM4` | 0.01052 |
| 34 | `SDLONELY` | 0.01038 |
| 35 | `LSATISFY` | 0.01004 |
| 36 | `_DRNKWK3` | 0.00984 |
| 37 | `_HCVU654` | 0.00980 |
| 38 | `LCSFIRST` | 0.00960 |
| 39 | `_AGE_G` | 0.00955 |
| 40 | `LCSLAST_` | 0.00943 |
| 41 | `_LCSYSMK` | 0.00935 |
| 42 | `_AGE65YR` | 0.00934 |
| 43 | `INCOME3` | 0.00912 |
| 44 | `HTIN4` | 0.00876 |
| 45 | `MARITAL` | 0.00866 |
| 46 | `LCSNUMCG` | 0.00825 |
| 47 | `_PACKDAY` | 0.00816 |
| 48 | `LCSNUMC_` | 0.00816 |
| 49 | `_LCSAGE` | 0.00802 |
| 50 | `CELLSEX3` | 0.00796 |
| 51 | `PRIMINS2` | 0.00786 |
| 52 | `CRVCLCNC` | 0.00779 |
| 53 | `_INCOMG1` | 0.00730 |
| 54 | `HAVARTH4` | 0.00681 |
| 55 | `_DRDXAR2` | 0.00681 |
| 56 | `DIFFWALK` | 0.00667 |
| 57 | `_BMI5CAT` | 0.00655 |
| 58 | `EMTSUPRT` | 0.00622 |
| 59 | `_LCSSMKG` | 0.00612 |
| 60 | `CHCCOPD3` | 0.00592 |
| 61 | `_SMOKER3` | 0.00589 |
| 62 | `SSBSUGR2` | 0.00583 |
| 63 | `SSBFRUT3` | 0.00581 |
| 64 | `RENTHOM1` | 0.00570 |
| 65 | `HADMAM` | 0.00561 |
| 66 | `SOFEMALE` | 0.00534 |
| 67 | `DIFFDRES` | 0.00533 |
| 68 | `MARIJAN1` | 0.00532 |
| 69 | `CNCRAGE` | 0.00522 |
| 70 | `_CURECI3` | 0.00486 |
| 71 | `SDHFOOD1` | 0.00480 |
| 72 | `_RFSMOK3` | 0.00470 |
| 73 | `_MRACE1` | 0.00465 |
| 74 | `_RACE_enc` | 0.00458 |
| 75 | `SMOKDAY2` | 0.00436 |
| 76 | `_CRVSCRN` | 0.00435 |
| 77 | `SDHUTILS` | 0.00422 |
| 78 | `CDHOUS1` | 0.00417 |
| 79 | `USEMRJN4` | 0.00406 |
| 80 | `DROCDY4_` | 0.00406 |
| 81 | `SMOKE100` | 0.00392 |
| 82 | `_HPV5YR1` | 0.00383 |
| 83 | `_RACEPRV` | 0.00379 |
| 84 | `FLSHTMY3` | 0.00368 |
| 85 | `QSTVER` | 0.00362 |
| 86 | `_DUALUSE` | 0.00360 |
| 87 | `_FLSHOT7` | 0.00357 |
| 88 | `SDHBILLS` | 0.00356 |
| 89 | `TYPCNTR9` | 0.00353 |
| 90 | `_RACEGR3` | 0.00352 |
| 91 | `ALCDAY4` | 0.00345 |
| 92 | `HHADULT` | 0.00345 |
| 93 | `HOWLONG` | 0.00344 |
| 94 | `DRNK3GE5` | 0.00339 |
| 95 | `_IMPRACE` | 0.00337 |
| 96 | `_PNEUMO3` | 0.00334 |
| 97 | `SDHTRNSP` | 0.00330 |
| 98 | `LASTDEN4` | 0.00328 |
| 99 | `RMVTETH4` | 0.00319 |
| 100 | `MAXDRNKS` | 0.00310 |
| 101 | `PERSDOC3` | 0.00306 |
| 102 | `FOODSTMP` | 0.00295 |
| 103 | `CIMEMLO1` | 0.00291 |
| 104 | `_RAWRAKE` | 0.00281 |
| 105 | `LANDSEX3` | 0.00275 |
| 106 | `IDAY` | 0.00268 |
| 107 | `MARJVAPE` | 0.00264 |
| 108 | `_RACEG21` | 0.00249 |
| 109 | `_DENVST3` | 0.00245 |
| 110 | `CRGVPRB4` | 0.00240 |
| 111 | `_RFPAP37` | 0.00234 |
| 112 | `_MAM402Y` | 0.00231 |
| 113 | `BLIND` | 0.00224 |
| 114 | `QSTLANG` | 0.00223 |
| 115 | `HADHYST2` | 0.00222 |
| 116 | `CDSOCIA1` | 0.00218 |
| 117 | `LASTSMK2` | 0.00201 |
| 118 | `HOWSAFE1` | 0.00199 |
| 119 | `CRGVREL5` | 0.00197 |
| 120 | `CDDISCU1` | 0.00197 |
| 121 | `CPDEMO1C` | 0.00194 |
| 122 | `EDUCA` | 0.00193 |
| 123 | `MARJEAT` | 0.00192 |
| 124 | `LCSCTSC1` | 0.00190 |
| 125 | `AVEDRNK4` | 0.00183 |
| 126 | `ASTHNOW` | 0.00182 |
| 127 | `_RFBMI5` | 0.00179 |
| 128 | `DIABETE4` | 0.00178 |
| 129 | `SOMALE` | 0.00173 |
| 130 | `_EDUCAG` | 0.00170 |
| 131 | `_EXTETH3` | 0.00168 |
| 132 | `PCSTALK2` | 0.00167 |
| 133 | `HIVRISK5` | 0.00164 |
| 134 | `_LCSCTSN` | 0.00163 |
| 135 | `CHCKDNY2` | 0.00153 |
| 136 | `SDHEMPLY` | 0.00151 |
| 137 | `ACETTHEM` | 0.00151 |
| 138 | `PSATEST1` | 0.00149 |
| 139 | `MSCODE` | 0.00143 |
| 140 | `HPVADSH1` | 0.00143 |
| 141 | `MARJSMOK` | 0.00140 |
| 142 | `CRGVNURS` | 0.00140 |
| 143 | `LANDLINE` | 0.00140 |
| 144 | `EXERANY2` | 0.00139 |
| 145 | `NOBCUSE8` | 0.00138 |
| 146 | `ACEDEPRS` | 0.00138 |
| 147 | `VCLNTES2` | 0.00138 |
| 148 | `CASTHDX2` | 0.00130 |
| 149 | `_PAPHPV1` | 0.00129 |
| 150 | `IMONTH` | 0.00128 |
| 151 | `FMONTH` | 0.00127 |
| 152 | `CNCRTYP2` | 0.00127 |
| 153 | `DIABEDU1` | 0.00126 |
| 154 | `_CHISPNC` | 0.00125 |
| 155 | `FEETSORE` | 0.00124 |
| 156 | `_TOTINDA` | 0.00123 |
| 157 | `_HISPANC` | 0.00123 |
| 158 | `HPVADVC4` | 0.00122 |
| 159 | `STOPSMK2` | 0.00121 |
| 160 | `IMFVPLA5` | 0.00114 |
| 161 | `VETERAN3` | 0.00114 |
| 162 | `COLNTES1` | 0.00113 |
| 163 | `CHCSCNC1` | 0.00112 |
| 164 | `_CRACE1` | 0.00111 |
| 165 | `CHILDREN` | 0.00107 |
| 166 | `LCSCTWHN` | 0.00103 |
| 167 | `PREDIAB2` | 0.00102 |
| 168 | `HADSIGM4` | 0.00100 |
| 169 | `_HADCOLN` | 0.00096 |
| 170 | `ACEADSAF` | 0.00094 |
| 171 | `ACETOUCH` | 0.00086 |
| 172 | `CSRVDOC1` | 0.00086 |
| 173 | `CRVCLPAP` | 0.00083 |
| 174 | `CRGVLNG2` | 0.00080 |
| 175 | `_RFBING6` | 0.00080 |
| 176 | `_RFMAM23` | 0.00079 |
| 177 | `PFPPRVN4` | 0.00078 |
| 178 | `CRGVHRS2` | 0.00075 |
| 179 | `ACESWEAR` | 0.00074 |
| 180 | `ACEADNED` | 0.00073 |
| 181 | `ACEPUNCH` | 0.00073 |
| 182 | `CRGVALZD` | 0.00071 |
| 183 | `EYEEXAM1` | 0.00070 |
| 184 | `MENTCIGS` | 0.00070 |
| 185 | `_LCSPSTF` | 0.00068 |
| 186 | `CSRVCTL2` | 0.00067 |
| 187 | `SIGMTES1` | 0.00066 |
| 188 | `GUNLOAD` | 0.00066 |
| 189 | `FIREARM5` | 0.00065 |
| 190 | `SHINGLE2` | 0.00064 |
| 191 | `RCSRLTN2` | 0.00064 |
| 192 | `CRGVPER2` | 0.00064 |
| 193 | `ACEHURT1` | 0.00063 |
| 194 | `CAGEG` | 0.00063 |
| 195 | `NUMADULT` | 0.00060 |
| 196 | `CSRVTRT3` | 0.00059 |
| 197 | `HADSEX` | 0.00058 |
| 198 | `CHKHEMO3` | 0.00056 |
| 199 | `_LCSELIG` | 0.00056 |
| 200 | `LASTSIG4` | 0.00055 |
| 201 | `DRNKANY6` | 0.00055 |
| 202 | `HPVDSHT` | 0.00051 |
| 203 | `ACEHVSEX` | 0.00050 |
| 204 | `PNEUVAC4` | 0.00048 |
| 205 | `_CHLDCNT` | 0.00048 |
| 206 | `TETANUS1` | 0.00048 |
| 207 | `INSULIN1` | 0.00048 |
| 208 | `_CRCREC3` | 0.00047 |
| 209 | `SMALSTOL` | 0.00046 |
| 210 | `_ALTETH3` | 0.00045 |
| 211 | `CHCOCNC1` | 0.00044 |
| 212 | `CASTHNO2` | 0.00044 |
| 213 | `RESPSLC1` | 0.00044 |
| 214 | `_RFDRHV9` | 0.00044 |
| 215 | `MARJOTHR` | 0.00041 |
| 216 | `CDWORRY` | 0.00041 |
| 217 | `CSRVPAIN` | 0.00041 |
| 218 | `CNCRDIFF` | 0.00040 |
| 219 | `CHECKUP1` | 0.00040 |
| 220 | `CAREGIV1` | 0.00039 |
| 221 | `COLNSIGM` | 0.00039 |
| 222 | `PREGNANT` | 0.00038 |
| 223 | `PDIABTS1` | 0.00038 |
| 224 | `ICFQSTVR` | 0.00036 |
| 225 | `LCSSCNCR` | 0.00035 |
| 226 | `PCPSARS2` | 0.00034 |
| 227 | `DEAF` | 0.00034 |
| 228 | `PVTRESD3` | 0.00034 |
| 229 | `DIABEYE1` | 0.00032 |
| 230 | `NUMPHON4` | 0.00032 |
| 231 | `CVDSTRK3` | 0.00031 |
| 232 | `NUMHHOL4` | 0.00030 |
| 233 | `LOADULK2` | 0.00030 |
| 234 | `_MICHD` | 0.00030 |
| 235 | `_STOLDN2` | 0.00029 |
| 236 | `PSASUGS1` | 0.00026 |
| 237 | `ACEPRISN` | 0.00026 |
| 238 | `_CLNSCP2` | 0.00026 |
| 239 | `DISPCODE` | 0.00025 |
| 240 | `HEATTBCO` | 0.00024 |
| 241 | `PSATIME1` | 0.00023 |
| 242 | `SDNATES1` | 0.00023 |
| 243 | `_VIRCOL2` | 0.00023 |
| 244 | `CSRVSUM` | 0.00021 |
| 245 | `ACEDIVRC` | 0.00020 |
| 246 | `ACEDRUGS` | 0.00019 |
| 247 | `ARTHEXER` | 0.00019 |
| 248 | `MENTECIG` | 0.00019 |
| 249 | `LADULT1` | 0.00019 |
| 250 | `CSTATE1` | 0.00018 |
| 251 | `COLNCNCR` | 0.00017 |
| 252 | `CRGVHOU2` | 0.00017 |
| 253 | `STOLTEST` | 0.00016 |
| 254 | `USENOW3` | 0.00016 |
| 255 | `CVDINFR4` | 0.00014 |
| 256 | `_SGMSCP2` | 0.00014 |
| 257 | `BLDSTFIT` | 0.00013 |
| 258 | `_SGMS102` | 0.00012 |
| 259 | `CERVSCRN` | 0.00012 |
| 260 | `VIRCOLO1` | 0.00012 |
| 261 | `STOOLDN2` | 0.00012 |
| 262 | `CVDCRHD4` | 0.00010 |
| 263 | `_RFBLDS6` | 0.00009 |
| 264 | `CSRVCLIN` | 0.00009 |
| 265 | `CSRVRTRN` | 0.00008 |
| 266 | `CSRVINST` | 0.00008 |
| 267 | `CSRVDEIN` | 0.00008 |
| 268 | `_SBONTI2` | 0.00007 |
| 269 | `FLUSHOT7` | 0.00007 |
| 270 | `DIABTYPE` | 0.00006 |
| 271 | `CSRVINSR` | 0.00005 |
| 272 | `ACEDRINK` | 0.00003 |
| 273 | `_URBSTAT` | 0.00002 |
| 274 | `_METSTAT` | 0.00001 |
| 275 | `MARJDAB` | 0.00001 |
| 276 | `_HADSIGM` | 0.00000 |
| 277 | `_HLTHPL2` | 0.00000 |
| 278 | `IYEAR` | 0.00000 |
| 279 | `CELLFON5` | 0.00000 |
| 280 | `CTELNUM1` | 0.00000 |
| 281 | `CADULT1` | 0.00000 |
| 282 | `CCLGHOUS` | 0.00000 |
| 283 | `SAFETIME` | 0.00000 |
| 284 | `STATERE1` | 0.00000 |
| 285 | `CELPHON1` | 0.00000 |
| 286 | `CTELENM1` | 0.00000 |
| 287 | `PVTRESD1` | 0.00000 |
| 288 | `COLGHOUS` | 0.00000 |

## Permutation importance (RF, ROC-AUC drop, top-30 RF)

| variable | ROC-AUC drop |
|---|---|
| `MENTHLTH` | 0.03593 |
| `_MENT14D` | 0.02084 |
| `POORHLTH` | 0.00444 |
| `DECIDE` | 0.01438 |
| `_PHYS14D` | -0.00003 |
| `PHYSHLTH` | 0.00014 |
| `GENHLTH` | 0.00108 |
| `SDLONELY` | 0.00223 |
| `DIFFALON` | 0.00071 |
| `_RFHLTH` | 0.00007 |
| `_SEX` | 0.00154 |
| `SEXVAR` | 0.00189 |
| `ECIGNOW3` | 0.00211 |
| `_AGE80` | 0.00169 |
| `EMPLOY1` | 0.00085 |
| `ASTHMA3` | 0.00049 |
| `_ASTHMS1` | 0.00055 |
| `_AGEG5YR` | 0.00102 |
| `_BMI5` | 0.00066 |
| `LSATISFY` | 0.00068 |
| `HIVTST7` | 0.00081 |
| `_LTASTH1` | 0.00040 |
| `_DRDXAR2` | 0.00163 |
| `CELLSEX3` | 0.00078 |
| `_AIDTST4` | 0.00071 |
| `HIVTSTD3` | 0.00022 |
| `HEIGHT3` | 0.00076 |
| `HAVARTH4` | 0.00167 |
| `WTKG3` | 0.00054 |
| `_STSTR` | 0.00049 |
| `_RACE_enc` | nan |
| `_CASTHM1` | nan |
| `_AGE_G` | nan |
| `HTM4` | nan |
| `HTIN4` | nan |
| `_PACKYRS` | nan |
| `INCOME3` | nan |
| `_HCVU654` | nan |
| `_LCSYSMK` | nan |
| `MARITAL` | nan |
| `IDAY` | nan |
| `_LCSAGE` | nan |
| `_AGE65YR` | nan |
| `LCSFIRST` | nan |
| `RENTHOM1` | nan |
| `DIFFWALK` | nan |
| `MEDCOST1` | nan |
| `PRIMINS2` | nan |
| `_LCSSMKG` | nan |
| `_RACEGR3` | nan |
| `CRVCLCNC` | nan |
| `_SMOKER3` | nan |
| `_DUALCOR` | nan |
| `_RACEPRV` | nan |
| `EMTSUPRT` | nan |
| `CHCCOPD3` | nan |
| `IMONTH` | nan |
| `_IMPRACE` | nan |
| `FMONTH` | nan |
| `_INCOMG1` | nan |
| `_DRNKWK3` | nan |
| `_RACEG21` | nan |
| `LCSLAST_` | nan |
| `SMOKE100` | nan |
| `_LCSYQTS` | nan |
| `QSTVER` | nan |
| `_MRACE1` | nan |
| `PERSDOC3` | nan |
| `DROCDY4_` | nan |
| `FLSHTMY3` | nan |
| `LCSNUMCG` | nan |
| `ALCDAY4` | nan |
| `_BMI5CAT` | nan |
| `DIABAGE4` | nan |
| `_PACKDAY` | nan |
| `LCSNUMC_` | nan |
| `MAXDRNKS` | nan |
| `HHADULT` | nan |
| `EDUCA` | nan |
| `CRVCLHPV` | nan |
| `_EDUCAG` | nan |
| `COLNTES1` | nan |
| `_DUALUSE` | nan |
| `AVEDRNK4` | nan |
| `_PNEUMO3` | nan |
| `RMVTETH4` | nan |
| `_FLSHOT7` | nan |
| `FLUSHOT7` | nan |
| `IMFVPLA5` | nan |
| `CHECKUP1` | nan |
| `PNEUVAC4` | nan |
| `_LCSCTSN` | nan |
| `LASTDEN4` | nan |
| `_CHLDCNT` | nan |
| `QSTLANG` | nan |
| `LCSCTSC1` | nan |
| `SDHFOOD1` | nan |
| `_CRVSCRN` | nan |
| `SSBSUGR2` | nan |
| `SOFEMALE` | nan |
| `HOWLONG` | nan |
| `SSBFRUT3` | nan |
| `CIMEMLO1` | nan |
| `MARIJAN1` | nan |
| `CHILDREN` | nan |
| `HADMAM` | nan |
| `_RFBMI5` | nan |
| `HOWSAFE1` | nan |
| `HADHYST2` | nan |
| `DRNK3GE5` | nan |
| `PDIABTS1` | nan |
| `DIFFDRES` | nan |
| `CPDEMO1C` | nan |
| `EXERANY2` | nan |
| `HIVRISK5` | nan |
| `HADSIGM4` | nan |
| `DIABETE4` | nan |
| `MSCODE` | nan |
| `_EXTETH3` | nan |
| `_RAWRAKE` | nan |
| `_CURECI3` | nan |
| `_DENVST3` | nan |
| `_TOTINDA` | nan |
| `SMOKDAY2` | nan |
| `_CRCREC3` | nan |
| `_HADCOLN` | nan |
| `FOODSTMP` | nan |
| `_HISPANC` | nan |
| `LCSSCNCR` | nan |
| `STOLTEST` | nan |
| `BLIND` | nan |
| `NUMADULT` | nan |
| `_CLNSCP2` | nan |
| `SDHEMPLY` | nan |
| `DEAF` | nan |
| `LASTSMK2` | nan |
| `LCSCTWHN` | nan |
| `COLNCNCR` | nan |
| `_RFBING6` | nan |
| `LANDLINE` | nan |
| `SIGMTES1` | nan |
| `_RFBLDS6` | nan |
| `_RFDRHV9` | nan |
| `_SBONTI2` | nan |
| `_METSTAT` | nan |
| `CHCOCNC1` | nan |
| `_CHISPNC` | nan |
| `SOMALE` | nan |
| `SDHTRNSP` | nan |
| `DRNKANY6` | nan |
| `LANDSEX3` | nan |
| `CRGVPRB4` | nan |
| `CAGEG` | nan |
| `CHCKDNY2` | nan |
| `VETERAN3` | nan |
| `_RFSMOK3` | nan |
| `COLNSIGM` | nan |
| `FIREARM5` | nan |
| `_MICHD` | nan |
| `CNCRAGE` | nan |
| `_CRACE1` | nan |
| `CERVSCRN` | nan |
| `ASTHNOW` | nan |
| `_RFMAM23` | nan |
| `_ALTETH3` | nan |
| `SDHUTILS` | nan |
| `CVDCRHD4` | nan |
| `SDHBILLS` | nan |
| `DISPCODE` | nan |
| `PREDIAB2` | nan |
| `CRGVREL5` | nan |
| `CHCSCNC1` | nan |
| `_URBSTAT` | nan |
| `USENOW3` | nan |
| `TYPCNTR9` | nan |
| `CNCRTYP2` | nan |
| `_HPV5YR1` | nan |
| `_SGMSCP2` | nan |
| `_LCSPSTF` | nan |
| `ACESWEAR` | nan |
| `CHKHEMO3` | nan |
| `CVDSTRK3` | nan |
| `MARJEAT` | nan |
| `CDHOUS1` | nan |
| `CVDINFR4` | nan |
| `RCSRLTN2` | nan |
| `_SGMS102` | nan |
| `HEATTBCO` | nan |
| `CRGVLNG2` | nan |
| `_HADSIGM` | nan |
| `ACEDEPRS` | nan |
| `ACEADSAF` | nan |
| `ACETOUCH` | nan |
| `DIABEYE1` | nan |
| `_MAM402Y` | nan |
| `MARJVAPE` | nan |
| `VCLNTES2` | nan |
| `VIRCOLO1` | nan |
| `_HLTHPL2` | nan |
| `EYEEXAM1` | nan |
| `CRVCLPAP` | nan |
| `ACEHURT1` | nan |
| `CAREGIV1` | nan |
| `DIABEDU1` | nan |
| `SMALSTOL` | nan |
| `SDNATES1` | nan |
| `GUNLOAD` | nan |
| `STOOLDN2` | nan |
| `IYEAR` | nan |
| `CSTATE1` | nan |
| `_RFPAP37` | nan |
| `_STOLDN2` | nan |
| `CDSOCIA1` | nan |
| `ACEPUNCH` | nan |
| `TETANUS1` | nan |
| `ACEDIVRC` | nan |
| `USEMRJN4` | nan |
| `ACEADNED` | nan |
| `ACEDRINK` | nan |
| `_LCSELIG` | nan |
| `PCSTALK2` | nan |
| `CRGVHRS2` | nan |
| `ARTHEXER` | nan |
| `MARJSMOK` | nan |
| `CDDISCU1` | nan |
| `CASTHDX2` | nan |
| `_PAPHPV1` | nan |
| `CDWORRY` | nan |
| `PSATEST1` | nan |
| `STOPSMK2` | nan |
| `MENTCIGS` | nan |
| `ACETTHEM` | nan |
| `BLDSTFIT` | nan |
| `_VIRCOL2` | nan |
| `CRGVPER2` | nan |
| `ACEDRUGS` | nan |
| `ACEHVSEX` | nan |
| `PSATIME1` | nan |
| `LOADULK2` | nan |
| `NUMHHOL4` | nan |
| `HPVADVC4` | nan |
| `CRGVHOU2` | nan |
| `LASTSIG4` | nan |
| `HADSEX` | nan |
| `SHINGLE2` | nan |
| `ACEPRISN` | nan |
| `INSULIN1` | nan |
| `DIABTYPE` | nan |
| `NOBCUSE8` | nan |
| `MENTECIG` | nan |
| `CRGVNURS` | nan |
| `NUMPHON4` | nan |
| `CRGVALZD` | nan |
| `CSRVDOC1` | nan |
| `CSRVTRT3` | nan |
| `CNCRDIFF` | nan |
| `FEETSORE` | nan |
| `HPVADSH1` | nan |
| `PFPPRVN4` | nan |
| `PCPSARS2` | nan |
| `HPVDSHT` | nan |
| `PREGNANT` | nan |
| `MARJOTHR` | nan |
| `MARJDAB` | nan |
| `CASTHNO2` | nan |
| `RESPSLC1` | nan |
| `CSRVRTRN` | nan |
| `PVTRESD3` | nan |
| `CSRVSUM` | nan |
| `PSASUGS1` | nan |
| `CSRVPAIN` | nan |
| `CSRVDEIN` | nan |
| `CSRVCTL2` | nan |
| `ICFQSTVR` | nan |
| `CSRVINSR` | nan |
| `CSRVINST` | nan |
| `CSRVCLIN` | nan |
| `LADULT1` | nan |
| `PVTRESD1` | nan |
| `CCLGHOUS` | nan |
| `COLGHOUS` | nan |
| `CTELENM1` | nan |
| `CADULT1` | nan |
| `CELLFON5` | nan |
| `STATERE1` | nan |
| `CELPHON1` | nan |
| `CTELNUM1` | nan |
| `SAFETIME` | nan |

## Overlap RF ∩ MI

25 / 30 compartidas entre el top-30 de RF y el top-30 de MI.

## Set canonico (top-30 RF)

```
`MENTHLTH`, `_MENT14D`, `POORHLTH`, `DECIDE`, `_PHYS14D`, `PHYSHLTH`, `GENHLTH`, `SDLONELY`, `DIFFALON`, `_RFHLTH`, `_SEX`, `SEXVAR`, `ECIGNOW3`, `_AGE80`, `EMPLOY1`, `ASTHMA3`, `_ASTHMS1`, `_AGEG5YR`, `_BMI5`, `LSATISFY`, `HIVTST7`, `_LTASTH1`, `_DRDXAR2`, `CELLSEX3`, `_AIDTST4`, `HIVTSTD3`, `HEIGHT3`, `HAVARTH4`, `WTKG3`, `_STSTR`
```

## Notas

- El canonico es el top-30 de RF. MI sirve de contraste; tiende a
  contaminarse con pesos de diseno y variables estructurales, por eso
  no se usa como set final.
- `_RACE` se mean-encodea aparte en el pipeline (no entra en las 30).
- El preset `default` en `brfss_dataset_model.py` aplica dos sustituciones
  manuales sobre este top-30: quita `_MENT14D` (casi-duplicado del
  target) y `_STSTR` (estrato de muestreo), y agrega `_PACKYRS` y
  `MEDCOST1` del rango RF siguiente.
