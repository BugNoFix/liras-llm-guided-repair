# Reference per lo stato dell'arte

Data di raccolta: 2026-07-11

Contesto del progetto: pipeline per generare modelli LIRAs da scenari in linguaggio naturale usando LLM, ripararli con feedback del compilatore, esportarli in XML e validarli con query UPPAAL/verifyta.

## Lettura rapida

Per la tesi, il progetto si colloca all'incrocio di quattro linee di ricerca:

1. generazione NL-to-DSL e model-driven engineering con LLM;
2. generazione di specifiche formali e logiche temporali da linguaggio naturale;
3. repair iterativo guidato da errori, test, parser, compilatori o model checker;
4. validazione formale di sistemi real-time/multi-agent con UPPAAL e timed automata.

Le reference piu' centrali per il tuo stato dell'arte sono: Text2DSL, From Text to DSL, Can LLMs Write Correct TLA+ Specifications?, Syntax Is Easy, Semantics Is Hard, Lang2LTL, AutoSafeLTL, SpecVerify, Self-Debugging, PracRepair e UPPAAL.

## A. NL-to-DSL e DSL generation



### Pro1. Text2DSL: LLM-Based Code Generation for Domain-Specific Languages

- Autori: Alexander V. Kozachok, Alexander M. Nazimov, Shamil G. Magomedov
- Anno: 2026
- Link: [https://arxiv.org/abs/2606.22586](https://arxiv.org/abs/2606.22586)
- Tipo: preprint arXiv
- Pertinenza: molto alta
- Da usare per: definire il problema NL-to-DSL come classe distinta da text-to-code generico.
- Perche' e' utile: formalizza il mapping da descrizioni naturali a programmi in DSL con grammatica BNF, vocabolario chiuso e controlli strutturali/AST. E' molto vicino al tuo uso di prompt con grammatica LIRAs, vincoli semantici e validazione tramite compilatore.
- Nota critica: e' un preprint recente; va citato come lavoro emergente, non come riferimento consolidato.



### Pro2. From Text to DSL: Evaluating Grammar-Based Model Generation Using Open LLMs

- Autori: Junaid Baber, Nicolas Hili, Didier Schwab, Leo Challier, Cecilia Satrin
- Anno: 2026
- Link: [https://arxiv.org/abs/2605.15865](https://arxiv.org/abs/2605.15865)
- Tipo: preprint arXiv
- Pertinenza: molto alta
- Da usare per: motivare l'uso di modelli open-weight nella generazione di modelli conformi a DSL.
- Perche' e' utile: valuta 39 LLM open-source per generare modelli conformi a DSL da linguaggio naturale, usando parsing automatico ed expert feedback. E' una buona base per confrontare sintassi, completezza semantica e consistenza dei riferimenti.
- Nota critica: non tratta LIRAs ne' UPPAAL, ma il setting e' direttamente analogo.



## B. Specifiche formali e logiche temporali generate da LLM



### Pro6. Syntax Is Easy, Semantics Is Hard: Evaluating LLMs for LTL Translation 

- Autori: Priscilla Kyei Danso, Mohammad Saqib Hasan, Niranjan Balasubramanian, Omar Chowdhury
- Anno: 2026
- Link: [https://arxiv.org/abs/2604.07321](https://arxiv.org/abs/2604.07321)
- Tipo: paper ACM SecDev 2026, arXiv
- Pertinenza: molto alta
- Da usare per: distinguere sintassi valida da correttezza semantica.
- Perche' e' utile: mostra che gli LLM possono produrre formule LTL sintatticamente plausibili ma semanticamente errate. Questo e' molto importante per spiegare perche' nel tuo progetto non basta compilare il DSL: serve anche validazione UPPAAL.



### MID9. Automatic Generation of Safety-compliant Linear Temporal Logic via Large Language Model: A Self-supervised Framework

- Autori: Junle Li, Siqi Chen, Jiakai Li, Meiqi Tian, Bingzhuo Zhong
- Anno: 2025, revisionato 2026
- Link: [https://arxiv.org/abs/2503.15840](https://arxiv.org/abs/2503.15840)
- Tipo: preprint arXiv
- Pertinenza: alta
- Da usare per: feedback automatico e counterexample-guided correction.
- Perche' e' utile: AutoSafeLTL combina LLM, language inclusion check e meccanismi di modifica guidati da controesempi per ottenere LTL safety-compliant.
- Collegamento al progetto: e' concettualmente vicino al tuo loop: generazione, controllo formale, feedback, nuova generazione.



### mid10. Translating Natural Language to Strategic Temporal Specifications via LLMs

- Autori: Marco Aruta, Francesco Improta, Vadim Malvone, Aniello Murano, Vladana Perlic
- Anno: 2026
- Link: [https://arxiv.org/abs/2606.30441](https://arxiv.org/abs/2606.30441)
- Tipo: preprint arXiv
- Pertinenza: alta
- Da usare per: multi-agent systems e requisiti temporali/strategici.
- Perche' e' utile: traduce descrizioni NL in formule ATL/ATL*, crea dataset validato da esperti e integra il sistema con un model checker.
- Nota critica: recentissimo e preprint; citarlo come lavoro emergente.



### mid11. ViTL: Temporal Logic-Guided Zero-Shot Natural Language Navigation via Vision-Language Models

- Autori: Kaier Liang, Hengde Dai, Cristian-Ioan Vasile
- Anno: 2026
- Link: [https://arxiv.org/abs/2606.30696](https://arxiv.org/abs/2606.30696)
- Tipo: preprint arXiv
- Pertinenza: media-alta
- Da usare per: applicazioni robotiche recenti con NL, LTL e navigazione long-horizon.
- Perche' e' utile: usa LLM per compilare comandi naturali in LTL, poi converte in DFA per coordinare sottotask e replanning.
- Nota critica: include visione e navigazione, quindi e' piu' applicativo e meno DSL/formal-methods puro.



## C. LLM + formal verification / model checking

.



## D. Repair, self-debugging e feedback loop



### MID15. Teaching Large Language Models to Self-Debug

- Autori: Xinyun Chen, Maxwell Lin, Nathanael Schaerli, Denny Zhou
- Anno: 2023
- Link: [https://arxiv.org/abs/2304.05128](https://arxiv.org/abs/2304.05128)
- Tipo: preprint arXiv
- Pertinenza: molto alta
- Da usare per: fondare il concetto di self-debugging e riuso di feedback.
- Perche' e' utile: mostra che gli LLM possono correggere predizioni precedenti usando few-shot demonstrations, spiegazioni ed execution feedback.
- Collegamento al progetto: e' una reference chiave per il tuo loop di repair su errori del compilatore LIRAs.

