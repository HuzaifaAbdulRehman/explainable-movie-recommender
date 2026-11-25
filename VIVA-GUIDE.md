# Explainable Movie Recommendations with QUBO Reranking

Roman Urdu viva guide. Pehli dafa parh rahe ho to sections 1–5 se start karo. Phir QUBO aur evaluation par aao. Aakhir mein demo chala kar apne alfaaz mein flow explain karo.

## 1. Project kahan se shuru hua?

Course proposal ne ek sawal uthaya. Agar recommender sirf highest relevance scores uthaye, to list ek hi tarah ki movies se bhar sakti hai. Kya hum relevance ka kuch hissa trade karke recommendation list mein variety aur genres ka mix improve kar sakte hain? Aur kya hum user ko dikha sakte hain ke koi movie kyon aayi?

Is idea ka background **feasible-rerank** hai. Local folder ka naam `qubo-rerank` hai aur uske README ka project title `feasible-rerank` hai. Course draft bhi existing ItemKNN/QUBO components ka zikr karta hai. Dono mein common idea hai ke relevance score pehle milta hai, phir quadratic objective poori list ko dekh kar items select karta hai. Feasible swap ka idea bhi common hai. Course app apne `recommender.py` se run hoti hai. Git history mein direct file-by-file copy ka record establish nahi hua, is liye viva mein **“we adapted the list-selection idea from feasible-rerank”** kehna accurate hai.

| Feasible-rerank ka research focus | Is course app ka focus |
|---|---|
| Relevance, diversity aur category/provider exposure par research experiments | Movies ki relevance, aapas ki similarity aur genres ka mix |
| Mukhtalif solver variants aur technical comparisons | Ek CPU swap annealer aur teen baselines, sab ek candidate pool par |
| Energy cost aur multiple benchmarks ka broader kaam | MovieLens 100K par ranking metrics aur waqt milliseconds mein |
| Optimization aur experiments ki research codebase | Browser demo, do sliders, explanations aur New user input |

**Timeline.** Proposal mein MovieLens, ItemKNN, QUBO, baselines aur evaluation plan kiye gaye. Pehle runnable version ne dataset load, similarity, browser demo, tests aur report add ki. Phir New user mode add hua. Uske baad UI do views mein gayi, Recommend aur Evaluation. Selected movies ko visible removable chips bana diya gaya.

Sources: [feasible-rerank](https://github.com/HuzaifaAbdulRehman/feasible-rerank) and the [course report](Report.pdf).

## 2. Recommender, input aur output

Recommender ka kaam user ko aisi movies suggest karna hai jo uski available history se match kar sakti hain. Hamara app do tarah se history leta hai:

- **MovieLens mode.** User ID select karo. App us ID ki training movies load karta hai. User 1 ke liye demo mein 271 history movies hain.
- **New user mode.** Catalog se 3–20 alag watched-and-liked movies choose karo. Ye titles current request ki history ban jaate hain. Account ya rating profile save karni zaroori nahi.
- **Diversity slider.** Similar movies ko ek list mein saath rakhne ki penalty control karta hai. MMR aur QUBO is weight ko use karte hain.
- **Genre balance slider.** QUBO mein target genre mix se deviation ki penalty control karta hai. Genre quota baseline ki apni fixed cap hai.

Output chaar methods ki ranked lists hain, har ek mein **10 alag movies**. History mein jo titles diye the woh output se exclude hote hain. Har result mein title, genres, normalized relevance aur history titles ke through explanation milti hai. Default view Top-k aur QUBO dikhata hai. MMR aur Genre quota expandable section mein hain.

## 3. MovieLens 100K aur test split

MovieLens 100K **GroupLens ka dataset** hai. Current archive mein **100,000 rating records**, **943 users** aur **1,682 movies** hain. `u.item` se movie ID, title aur genres aate hain. `u.data` ki har row mein user ID, movie ID, stars aur timestamp hota hai.

| Movie | Kab rate ki | Stars |
|---|---|---:|
| Toy Story | Monday | 5 |
| Aladdin | Tuesday | 4 |
| Batman | Wednesday | 2 |

**Batman test movie hai**, kyun ke Wednesday sabse baad hai. 2 stars select karne ki wajah nahi hain. Code timestamp descending sort karta hai. Equal timestamp ho to larger movie ID pehle aati hai. Har user se ek test record alag rakhne ke baad **99,057 training interactions** bachti hain. Test record similarity matrix aur user ki training history mein enter nahi hota.

**Implicit feedback** ka matlab yahan ye hai ke recorded rating ko interaction maana. Training matrix mein movie rate hui to entry 1, warna 0. Stars ki value scoring mein use nahi hoti. Is liye held-out Batman list mein aaye to hit count hogi, chahe user ne 2 stars diye hon. Ye protocol next recorded interaction ki retrieval measure karta hai. User ki actual pasand ko alag human feedback se dekh sakte hain.

Source: [GroupLens MovieLens 100K](https://grouplens.org/datasets/movielens/100k/).

## 4. ItemKNN candidate list kaise banata hai?

**Item** movie hai. **KNN** K Nearest Neighbors hai. Yahan neighbors woh history movies hain jo candidate movie se sabse zyada similar hain. Similarity ka signal common training users hain.

### Movie-to-movie similarity

Agar Batman ko 100 training users ne rate kiya, Batman Returns ko 80 ne, aur 40 users common hain, to:

- Cosine = 40 / sqrt(100 × 80) ≈ 0.447.
- Shrinkage = 40 / (40 + 10) = 0.8.
- Final similarity ≈ 0.358.

Cosine total popularity ko account karta hai. Shrinkage kam overlap wale pairs par score downweight karta hai. Do rare movies ko sirf ek common user ne rate kiya ho to raw cosine high aa sakta hai. Shrinkage us weak evidence ko chhota weight deta hai. Common users zero hon to similarity zero hai. Apni movie se similarity diagonal mein zero rakhi gayi hai.

Code binary user–movie matrix ko uske transpose ke saath multiply karta hai. Result mein har movie pair ka common-user count milta hai. Ye calculation server start par ek baar hoti hai. New user ki fresh choices isi trained similarity matrix se score hoti hain.

### User ke liye candidate score

Maan lo tumne Batman aur Jurassic Park select ki. System Batman Returns ka match in dono titles se dekhega aur scores add karega. Toy Story aur baaki unseen candidates ke scores bhi isi tarah nikalte hain.

1. Har candidate ke liye user ki history se **20 highest similarities** ka sum liya jata hai. Agar 3 favorites hain to teeno use hote hain.
2. History wali movies ko ranking se exclude kiya jata hai.
3. Scores descending sort karke **60 candidates** rakhe jaate hain.
4. Har candidate ka score pool ke highest score se divide hota hai. Isliye movie relevance screen par comparable within that request hoti hai. Ye predicted stars ya calibrated probability nahi hai.

**20, 60 aur 10.** 20 history neighbors per candidate. 60 movies reranker ko milti hain. 10 final list mein aati hain. Agar held-out movie 60 candidates mein hi nahi aayi, chaaron methods ke liye woh miss hai.

## 5. Chaar methods ek pool se 10 kaise chunte hain?

| Method | Selection ka rule | Batman wali misaal |
|---|---|---|
| **Top-k** | Pehle 10 highest relevance scores. k = 10. | Batman aur Batman Returns dono high score par aa sakti hain. |
| **MMR** | Ek ek movie select karo. Score = relevance − 2 × diversity × selected list ke saath maximum similarity. | Batman select ho chuki ho to Batman Returns ki penalty barh sakti hai aur Toy Story ko chance milta hai. |
| **Genre quota** | Relevance order mein next movie lo jisse kisi genre ki exposure 2 units cross na kare. Agar koi eligible na ho, sabse kam total excess wala candidate lo. | Action exposure bhar chuki ho to agla high-score Action title skip ho sakta hai. |
| **QUBO** | Poori 10-movie set ki relevance, pairwise similarity aur genre exposure ka combined objective minimize karo. | Batman Returns ko Toy Story se swap karo aur poori list ka score dobara compare karo. |

MMR greedy hai. Har step current list se best choice karta hai. Genre quota bhi sequential hai. QUBO selected set ka member badal kar alternatives explore karta hai. **Display ke liye chaaron methods selected movies ko ItemKNN relevance order mein dikhate hain.** MMR ka pick order aur screen rank is liye alag ho sakte hain.

## 6. QUBO aur feasible simulated annealing

QUBO ka full form **Quadratic Unconstrained Binary Optimization** hai. Binary variable xᵢ = 1 se movie i selected hai. xᵢ = 0 se woh bahar hai. Quadratic ka reason xᵢ × xⱼ hai. Agar dono movies selected hon to pair ki cost count hoti hai.

Course app ka selection objective is tarah samjho:

```text
cost = - selected movies ki total relevance
       + 2 × diversity weight × selected pairs ki total similarity
       + genre weight × squared genre exposure deviation
```

- Minus relevance ka matlab relevant movie objective ko neeche le jaati hai.
- Similarity penalty redundant pair ka cost barhati hai. Code symmetric pair matrix ki dono directions sum karta hai, isliye formula mein unique pairs par factor 2 hai.
- Genre target 10 exposure units ko 18 known genres mein evenly baantta hai. Har known genre ka target 10/18 hai. Unknown ka target zero hai.
- Action + Thriller movie exposure mein 0.5 Action aur 0.5 Thriller deti hai. Teen genres ho to har genre ko 1/3.
- Squared deviation ko expand karo to individual movie terms aur pair terms milte hain. Constant term candidate decisions badalti nahi, isliye solver usay drop karta hai.

**Unconstrained word ka viva sawaal.** QUBO ek binary quadratic objective ka naam hai. Is app mein **exactly 10** ka rule solver ke feasible search space mein enforce hota hai. Large cardinality penalty laga kar one-bit moves chalane ki zaroorat nahi padti. Feasible ka seedha matlab hai har proposal aur accepted list mein 10 titles rehte hain.

### Solver ke steps

1. Top-k ki 10 movies se start karo. Remaining 50 outside hain.
2. Ek selected aur ek outside candidate random choose karo. Swap propose karo. 10 ki count same rehti hai.
3. New cost minus old cost ko delta kahte hain. Code sirf swap se affected linear aur pair terms update karta hai.
4. Delta ≤ 0 ho to swap accept. Worse swap kabhi kabhi exp(−delta / temperature) probability se accept hoti hai. Is se search immediate local solution se bahar ja sakti hai.
5. 600 steps mein temperature geometrically 0.30 se 0.005 tak girta hai. Seed 42 repeatability mein help karta hai.
6. Search ke dauran mili **lowest-cost set** return karo. Screen us set ko relevance order mein dikhati hai.

Dono weights zero hon to objective sirf relevance reh jata hai. Search already Top-k se start karti hai aur best list track karti hai, isliye Top-k list retain hoti hai. Solver CPU par classical simulated annealing chalata hai. Quantum computer ya guaranteed global optimum ka claim report nahi karti.

## 7. Explanation, sliders aur genre breakdown

**Related to** ka matlab hai yeh history movies recommendation ke ItemKNN relevance score mein positive similarity contributors thin. Code top three positive contributors dikhata hai. Misal mein Beauty and the Beast ke neeche Aladdin, Snow White aur The Lion King aa sakti hain, agar woh current profile ki history mein hain. Lists alag profiles ya weights se badal sakti hain.

Movie-level explanation **score ka source** batati hai. QUBO **set-level selection** karta hai. Sirf Related to se yeh prove nahi hota ke kisi specific swap mein exactly kaunsa objective term decisive tha.

Diversity slider 0.45 ka matlab **45% diversity** nahi hai. Ye objective ka weight hai. Genre balance 0.30 bhi coefficient hai. Higher weight doosri cheezon ke saath trade-off badalta hai. Finite search aur fixed candidate pool ki wajah se har slider increment par metric ka monotonic change guarantee nahi.

## 8. Offline evaluation, Batman se samjho

Har user ki one held-out movie ke liye har method 10 titles banata hai. Phir user-level scores ka **943 users par mean** nikalta hai.

| Metric | Batman example | Kaunsi direction? |
|---|---|---|
| **Recall@10** | Held-out Batman list mein kahin bhi ho to 1, missing ho to 0. | Higher, held-out retrieval ke liye. |
| **NDCG@10** | Batman rank 1 par ho to 1, rank 2 par 0.63, rank 5 par 0.39, missing ho to 0. | Higher, early ranking ke liye. |
| **Similarity** | List mein Batman, Batman Returns aur Toy Story hon to inko aur baaki titles ko pair by pair compare karo. 10 movies ke 45 unique pairs ka mean. | Lower, kam redundancy ke liye. |
| **Genre gap** | 10 movies ki exposure sum karo. Action-heavy set uniform target se aur door ho sakti hai. | Lower, target mix ke qareeb jane ke liye. |
| **Runtime** | Candidate retrieval aur method ki list selection mein kitne milliseconds lage. | Lower, faster response ke liye. |

**NDCG formula.** Ek hi binary relevant test item hai. Agar rank r par aaye, 1 / log₂(r + 1). Ideal rank 1 ka score 1 hai, isliye normalized value yahi hai. Batman ke stars formula mein use nahi hote.

**Similarity vs relevance.** Relevance candidate aur **user ki history** ka match hai. List Similarity final **recommended movies ka aapas mein** match hai. Code symmetric matrix ka sum 90 ordered pairs se divide karta hai, jo 45 unique pairs ke mean ke barabar hai.

**Genre gap formula.** Har genre ka |actual exposure − target exposure| lo. Sab sum karke 20 se divide karo. 20 = 2 × list size. Ye normalized genre shares ka total variation distance hai. QUBO objective genre difference **square** karta hai. Display metric **absolute** difference use karti hai. Dono target ki taraf dekhte hain, lekin formulas alag hain.

**Runtime boundaries.** Shared retrieval ka time har method ke reported runtime mein include hai. Matrix build, explanation formatting aur browser/network us timer mein nahi. Milliseconds ko electricity ya joules na kaho.

### Actual results

Seed 42, diversity 0.45, genre balance 0.30. Numbers `results.json` se rounded hain.

| Method | Recall@10 | NDCG@10 | Similarity | Genre gap | Mean ms |
|---|---:|---:|---:|---:|---:|
| Top-k | 0.0594 | 0.0270 | 0.5724 | 0.4978 | 6.28 |
| MMR | **0.0785** | **0.0332** | 0.4781 | 0.4963 | 8.79 |
| Genre quota | 0.0530 | 0.0250 | 0.5694 | 0.4801 | 14.27 |
| QUBO | 0.0498 | 0.0237 | **0.3321** | **0.3502** | 26.65 |

MMR ne 943 mein se takreeban **74** held-out movies retrieve ki. QUBO ki similarity relative to Top-k takreeban **42% kam** aur genre gap **30% kam** tha. Saath hi QUBO ka Recall/NDCG kam tha aur runtime zyada. MMR held-out ranking ke liye strongest thi, Top-k speed ke liye, aur QUBO evaluated settings par list variety aur target match ke liye.

## 9. Human feedback ka istemal

Evaluation page ka **Record your judgment** form MovieLens profile par do choices leta hai. Most relevant ka matlab kaunsi list user ki pasand se zyada match lagi. Most varied ka matlab kaunsi list mein alag tarah ki movies zyada lagin. Tie choose ki ja sakti hai.

Save se UTC time, user ID, slider weights aur dono choices local `judgments.csv` mein append hote hain. Form ka scope opinions collect karna hai. Is feature se model weights update nahi hote. Existing offline benchmark bhi same rehta hai. Course report offline numbers report karti hai aur formal participant study ke results claim nahi karti. Kisi demo click ko research study ka result na samjho.

## 10. Code map aur request flow

| File | Kaam |
|---|---|
| [recommender.py](recommender.py) | ZIP parsing, train/test split, similarity, ItemKNN retrieval, four selections, explanations aur scores. |
| [serve.py](serve.py) | Python HTTP server. Catalog, users, recommendations, benchmark aur feedback handle karta hai. |
| [index.html](index.html) | HTML/CSS/JavaScript UI. Profile switch, title search, removable chips, sliders, comparison aur Evaluation. |
| [evaluate.py](evaluate.py) | Held-out movies check karta hai aur all-user mean calculate karta hai. |
| [results.json](results.json) | Stored benchmark jo browser ki Evaluation view dikhati hai. |
| [test_recommender.py](test_recommender.py) | Dataset counts, split, movie mapping, ten unique unseen picks, repeatability aur custom favorites checks. |
| [Report.tex](Report.tex) | Research report ka editable LaTeX source. |

Browser GET `/api/users` ya `/api/movies` se controls load karta hai. Recommend par `/api/recommend` ko existing user ID ya repeated `movie` IDs, diversity aur balance milte hain. Server input validate karke history banata hai aur Model.run_history call karta hai. Response mein history aur four methods aate hain. Browser JSON se cards render karta hai. Evaluation `/api/results` se stored results padhti hai. Feedback POST `/api/judgment` se save hota hai.

## 11. Viva demo aur commands

GitHub repo clone karne aur [setup](README.md) complete karne ke baad, repo folder mein PowerShell se:

```powershell
.\.venv\Scripts\python.exe serve.py
```

Browser mein http://127.0.0.1:8765 kholo. Already running ho to browser refresh karo.

1. **MovieLens → User 1**. User history khol kar batao ke ye training history hai.
2. Both sliders zero kar ke run karo. Top-k aur QUBO ka same output show karo.
3. Default 0.45 / 0.30 restore karo. QUBO ki list aur similarity/genre gap compare karo.
4. **New user** select karo. Toy Story (1995), Aladdin (1992), Beauty and the Beast (1991) search karke add karo. Selected chips dikhao, phir run karo. Related to mein selected titles dikhao.
5. **Compare MMR and Genre quota** kholo. Phir **Evaluation** par all-user table explain karo. Ek profile ka card score aur all-user benchmark ko mix na karo.

Tests aur benchmark:

```powershell
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe evaluate.py
```

Tests internal correctness check karte hain. Evaluation actual data run karke `results.json` overwrite karti hai. Printed timing laptop aur current load se badal sakti hai.

## 12. Professor ke common follow-ups

- **Harry Potter kyon nahi?** Catalog MovieLens 100K ka hai. Aisi movies jo archive mein listed nahi, search mein nahi aayengi. Newer dataset par shift karne se model build aur benchmark dobara chalayenge.
- **Why 60 candidates?** Reranking ka search manageable rakha aur comparisons mein same depth use ki. Is choice se retrieval ceiling bhi aati hai. Pool depth sweep is course benchmark mein report nahi hui.
- **Does QUBO beat all other methods?** Is run mein similarity aur genre gap par best hai. MMR ka Recall/NDCG best hai. Objective aur weight selection decide karti hai ke kis trade-off ko preference mile.
- **Same genre wali movies always similar?** Similarity common-user behavior se aati hai. Genre gap metadata se aata hai. Dono related ho sakte hain lekin signals alag hain.
- **Same result every time?** Same data, history, settings aur fixed seed par selection repeatable hona chahiye. Runtime repeatably identical nahi hota.
- **Why not call it a rating predictor?** App rating occurrence ko binary signal banati hai aur unseen titles rank karti hai. 1–5 stars predict karna objective mein nahi hai.
- **Is the split chronological?** Har user ke andar latest interaction test hai. Sab users ke liye ek common calendar cutoff nahi lagaya.
- **Kya QUBO winner prove ho gaya?** Solver best set found deta hai. Global optimum, universal user preference aur wider catalog par superiority establish karne ke liye aur experiments chahiye.

**30-second viva answer.** “We learn movie-to-movie similarity from MovieLens training interactions. ItemKNN retrieves 60 unseen candidates for a profile. Four methods select 10 movies from the same pool. QUBO uses feasible CPU swaps to trade off relevance, redundancy and genre exposure. We explain the scores with contributing history movies and evaluate all 943 users with a held-out interaction. MMR led held-out ranking, while QUBO gave the least similar and most genre-balanced lists at our measured settings.”
