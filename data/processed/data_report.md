# Concadia data report

- Raw records: **96918**
- Accepted: **94906**
- Excluded: **2012**

## Exclusions

- target_too_short: 1745
- placeholder_target: 146
- target_too_long: 121

## Accepted by split

- train: 76030 (raw 77534)
- validation: 9415 (raw 9693)
- test: 9461 (raw 9691)

## Flags among accepted records

- exact_duplicate: 17071
- exact_duplicate_cross_split: 3259
- near_duplicate: 17153
- near_duplicate_cross_split: 3351
- overlaps_test_image: 1277
- target_equals_caption: 8511
- target_in_caption: 1576
- target_in_paragraph: 270

## Duplicate audit

- Exact (MD5) groups: 8081 covering 20676 records; 1110 groups span splits
- Near (imagehash.phash identical (Hamming 0)) groups: 8138 covering 20854 records; 1141 groups span splits
- Train/validation records whose image also occurs in test are flagged `overlaps_test_image` and kept out of experiment subsets.

## Lengths (accepted)

- Target words: median 10, p95 40
- Paragraph words: median 95, p95 242

## Experiment subsets

Rule: accepted, not flagged for target leakage or test-image overlap, at least one of caption/paragraph, unique image within subset; seed=13

- train_5k: 5000
- pilot: 300
- tiny: 20
- val: 300
- milestone_100: 100
- test: 500

## Inspected examples (validation)

**concadia_35670** (List of Seattle Mariners team records)
- Caption: Ichiro Suzuki, the holder of five single-season batting records for the Mariners
- Target: Ichiro Suzuki in a Mariners uniform
- Paragraph (start): Single-season batting recordsStatisticPlayerRecordSeasonRef(s)Batting averageIchiro Suzuki.3722004Home runsKen Griffey, Jr.56*1997, 1998RBIKen Griffey, Jr.1471997RunsAlex Rodriguez1411996HitsIchiro Suzuki2622004SinglesIchiro Suzuki2252004DoublesAlex Rodriguez541996TriplesIchiro Suzuki122005Stolen basesHarold Reynolds601987At batsIchiro Suzuki7042004Slugging percentageKen Griffey, Jr..6741994Extra-base hitsKen Griffey, Jr.931997Total basesKen Griffey, Jr.3931997On-base percentageEdgar Martnez.4791995OPSEdgar Martnez1.1071995WalksEdgar Martnez1231996StrikeoutsMike Cameron1762002 ...

**concadia_79462** (List of D.C. United records and statistics)
- Caption: D.C. United players, including Ben Olsen and Luciano Emilio, look for a corner kick against Real Madrid
- Target: Six soccer players in black and five in white views from above look up for a moving soccer ball coming toward them.
- Paragraph (start): 5 times: Jaime Moreno: (1997), (1999), (2004), (2005), (2006) 4 times: Marco Etcheverry: (1996), (1997), (1998), (1999) 3 times: Christian Gmez: (2005), (2006), (2007) ...

**concadia_85986** (List of songs recorded by Pink Martini)
- Caption: Pink Martini's lead vocalist and songwriter China Forbes performing in 2006
- Target: A woman wearing a black-and-white printed dress, standing behind a microphone with her arms extended.
- Paragraph (start): The group's founder Thomas Lauderdale performing in 2008 Michael Feinstein provides vocals on the track "How Long Will It Last?", originally by Max Lief and Joseph Meyer. ...

**concadia_47921** (List of birds of Singapore)
- Caption: Collared kingfisher (Todiramphus chloris) - Nee Soon Forest
- Target: Collared kingfisher
- Paragraph (start): Kingfishers are medium-sized birds with large heads, long, pointed bills, short legs and stubby tails. Common kingfisher (Alcedo atthis) - C/WV - water edgesRufous-collared kingfisher (Actenoides concretus) - Ex/RBBlue-eared kingfisher (Alcedo meninting) - R/RB - BBNP, CCNR, HNP, NTL, Poyan, ...

**concadia_63586** (First Church in Albany (Reformed))
- Caption: The 1715 church
- Target: A worn illustration of a building with a tall peaked roof topped with a cupola
- Paragraph (start): Three years later the church began its mission to the Mohawks, the Native American tribe in the area. It continued for almost 50 years and baptized over 300 converts. In 1715 the church built a larger, hip-roofed stone church around ...
