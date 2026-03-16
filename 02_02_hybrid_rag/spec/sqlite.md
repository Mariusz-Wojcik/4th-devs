Skip to content
asg017
sqlite-vec
Repository navigation
Code
Issues
136
 (136)
Pull requests
31
 (31)
Agents
Actions
Projects
Security
Insights
Owner avatar
sqlite-vec
Public
asg017/sqlite-vec
Go to file
t
Name		
asg017
asg017
docs
a2dd24f
 · 
last year
.github
sponsor update
last year
benchmarks
benchmark updates
2 years ago
bindings
fix ncruces build
2 years ago
examples
node:sqlite sample
last year
scripts
always include notes, even on alpha releases
2 years ago
site
docs
last year
tests
fix segfault on invalid vec_each() input, fixes #163
last year
.gitignore
Metadata filtering (#124)
2 years ago
ARCHITECTURE.md
Metadata filtering (#124)
2 years ago
LICENSE-APACHE
add licenses
2 years ago
LICENSE-MIT
add licenses
2 years ago
Makefile
Metadata filtering (#124)
2 years ago
README.md
sponsor update
last year
SECURITY.md
add security doc
2 years ago
TODO
Metadata filtering (#124)
2 years ago
VERSION
v0.1.7-alpha.2
last year
reference.yaml
Remove vec_npy_each from default entrypoint and move to sqlite3_vec_n…
2 years ago
sqlite-dist.toml
update sqlite-dist
2 years ago
sqlite-vec.c
fix segfault on invalid vec_each() input, fixes #163
last year
sqlite-vec.h.tmpl
cleanup
2 years ago
test.sql
Metadata filtering (#124)
2 years ago
tmp-static.py
static updates
2 years ago
Repository files navigation
README
Apache-2.0 license
MIT license
Security
sqlite-vec


An extremely small, "fast enough" vector search SQLite extension that runs anywhere! A successor to sqlite-vss

Important

sqlite-vec is a pre-v1, so expect breaking changes!

Store and query float, int8, and binary vectors in vec0 virtual tables
Written in pure C, no dependencies, runs anywhere SQLite runs (Linux/MacOS/Windows, in the browser with WASM, Raspberry Pis, etc.)
Store non-vector data in metadata, auxiliary, or partition key columns
Mozilla Builders logo

sqlite-vec is a Mozilla Builders project, with additional sponsorship from  Fly.io ,  Turso,  SQLite Cloud, and  Shinkai. See the Sponsors section for more details.

Installing
See Installing sqlite-vec for more details.

Language	Install	More Info	
Python	pip install sqlite-vec	sqlite-vec with Python	PyPI
Node.js	npm install sqlite-vec	sqlite-vec with Node.js	npm
Ruby	gem install sqlite-vec	sqlite-vec with Ruby	Gem
Go	go get -u github.com/asg017/sqlite-vec/bindings/go	sqlite-vec with Go	Go Reference
Rust	cargo add sqlite-vec	sqlite-vec with Rust	Crates.io
Datasette	datasette install datasette-sqlite-vec	sqlite-vec with Datasette	Datasette
rqlite	rqlited -extensions-path=sqlite-vec.tar.gz	sqlite-vec with rqlite	rqlite
sqlite-utils	sqlite-utils install sqlite-utils-sqlite-vec	sqlite-vec with sqlite-utils	sqlite-utils
Github Release			GitHub tag (latest SemVer pre-release)
Sample usage
.load ./vec0

create virtual table vec_examples using vec0(
  sample_embedding float[8]
);

-- vectors can be provided as JSON or in a compact binary format
insert into vec_examples(rowid, sample_embedding)
  values
    (1, '[-0.200, 0.250, 0.341, -0.211, 0.645, 0.935, -0.316, -0.924]'),
    (2, '[0.443, -0.501, 0.355, -0.771, 0.707, -0.708, -0.185, 0.362]'),
    (3, '[0.716, -0.927, 0.134, 0.052, -0.669, 0.793, -0.634, -0.162]'),
    (4, '[-0.710, 0.330, 0.656, 0.041, -0.990, 0.726, 0.385, -0.958]');


-- KNN style query
select
  rowid,
  distance
from vec_examples
where sample_embedding match '[0.890, 0.544, 0.825, 0.961, 0.358, 0.0196, 0.521, 0.175]'
order by distance
limit 2;
/*
┌───────┬──────────────────┐
│ rowid │     distance     │
├───────┼──────────────────┤
│ 2     │ 2.38687372207642 │
│ 1     │ 2.38978505134583 │
└───────┴──────────────────┘
*/
Sponsors
Development of sqlite-vec is supported by multiple generous sponsors! Mozilla is the main sponsor through the new Builders project.

Mozilla Builders logo

sqlite-vec is also sponsored by the following companies:

Fly.io logo Turso logo SQLite Cloud logo Shinkai logo
As well as multiple individual supporters on Github sponsors!

If your company interested in sponsoring sqlite-vec development, send me an email to get more info: https://alexgarcia.xyz

See Also
sqlite-ecosystem, Maybe more 3rd party SQLite extensions I've developed
sqlite-rembed, Generate text embeddings from remote APIs like OpenAI/Nomic/Ollama, meant for testing and SQL scripts
sqlite-lembed, Generate text embeddings locally from embedding models in the .gguf format
About
A vector search SQLite extension that runs anywhere!

Topics
sqlite sqlite-extension
Resources
 Readme
License
 Apache-2.0, MIT licenses found
Security policy
 Security policy
 Activity
Stars
 6.9k stars
Watchers
 63 watching
Forks
 277 forks
Report repository
Releases 68
v0.1.6 - Metadata columns, Partition keys, and Auxiliary column support
Latest
on Nov 20, 2024
+ 67 releases
Packages
No packages published
Contributors
15
@asg017
@dleviminzi
@otoolep
@sheldonrobinson
@jimmystridh
@punkish
@giovannibenussi
@7flash
@ncruces
@himself65
@Adriankhl
@eltociear
@tom-pollak
@bholmesdev
@little-huang
Deployments
224
 github-pages last year
+ 223 deployments
Languages
C
60.2%
 
Python
31.1%
 
Makefile
2.1%
 
Rust
1.8%
 
TypeScript
1.6%
 
Vue
1.1%
 
Other
2.1%
Footer
© 2026 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Community
Docs
Contact
Manage cookies
Do not share my personal information
