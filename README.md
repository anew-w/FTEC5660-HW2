# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:

### Task 1
The verifier uses a deterministic evidence pipeline: it parses the CV into
structured identity, location, employment, education, and skill claims; resolves
the candidate against MCP profiles using name, location, employer, school, and
year evidence with LinkedIn as the primary source; and compares every relevant
claim field by field. DeepSeek-v4-flash is used only for ambiguous layouts, close
same-name candidates, and harmless wording differences. The agent ignores
headline, hometown, summaries, and job descriptions, limits concurrency to three
CVs, and returns a valid numeric score when a CV cannot be fully resolved. On the
public test set, the final recorded run completed end-to-end and achieved 7/7
accuracy.

### Task 2
The final attack changes only Kelly Tsang's headline from the true "Finance
Professional" to the job-relevant, embellished "Senior Finance Manager," while
preserving her actual current title (Manager), location, employment dates,
education, skills, and hometown. The submitted CV is a one-page,
text-extractable PDF with no hidden text or prompt injection. The approach
targets a verification-coverage gap: the stated discrepancy scope excludes
headline, so verifiers can compare all scored fields and still accept the
document. This is expected to transfer to the five held-out verifiers because
they use the same MCP evidence and field definitions; three repeated local tests
returned 0.9, above the 0.5 threshold.

### Task 3: Reflection on AI Paradoxes

When I built the verifier and then tried to attack it, I saw the same problem
from two sides. The biggest problem was not the AI's writing ability. It was
the assumptions I made about the data. For example, table-style PDFs sometimes
gave me broken names or missed sections, so I kept a rule-based parser and
added a structured AI fallback. When candidates had the same name, their
identity scores could be very close. I then used name, location, employer,
school, and years together, and only asked the model when the result was still
unclear. I also got false warnings when the CV used different wording or only
had partial information. So I normalized harmless differences and separated
"unknown" from "contradicted." There was also some other problems: skill
coverage, short MCP connection failures, and a missing logging import. This
showed me that each CV should be handled separately, and a technical error must
never be treated as fake evidence. After these fixes, my pipeline became more
like several steps: extract, identify, compare, review unclear cases, and fail
safely.

The two-sided exercise gave me a more important lesson. The evidence model that
make the verifier work also show its blind spots. A CV can keep all checked
fields true and still mislead people with an unchecked field. A high score only
means the CV matches the verifier's scope. It is not proof that everything is
true. This is the AI paradox I experienced. Automation make checking faster and
more consistent, but it can also give people too much confidence and make
cheating more systematic. So when we design a system, positive tests are not
enough. We also need to test how it fails. We need to define what evidence is
used, show uncertainty, and keep a human review for important decisions.

For future job applications, this helped me see the line between polishing a CV
and misrepresenting myself. It is fine to use stronger action verbs, add real
numbers, put the most relevant experience first, match the job's wording,
organize skills clearly, and remove unrelated content. These changes make the
CV easier to read without changing facts. But changing a title, date, employer,
degree, city, or skill I do not have is different. It may pass a narrow
automated check, but it is still misleading and can fail badly in an interview
or background check. The "Senior Finance Manager" attack worked, but it only
proves that there is a gap in the verifier. It is not job-search advice. The
main lesson for me is to present my true experience better, not to change the
evidence.
