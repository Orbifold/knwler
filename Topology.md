Topology (from the Greek words τόπος, 'place, location', and λόγος, 'study') is the branch of mathematics concerned with the properties of a geometric object that are preserved under continuous deformations, such as stretching, twisting, crumpling, and bending; that is, without closing holes, opening holes, tearing, gluing, or passing through itself.
A topological space is a set endowed with a structure, called a topology, which allows defining continuous deformation of subspaces, and, more generally, all kinds of continuity. Euclidean spaces, and, more generally, metric spaces are examples of topological spaces, as any distance or metric defines a topology. The deformations that are considered in topology are homeomorphisms and homotopies. A property that is invariant under such deformations is a topological property. The following are basic examples of topological properties: the dimension, which allows distinguishing between a line and a surface; compactness, which allows distinguishing between a line and a circle; connectedness, which allows distinguishing a circle from two non-intersecting circles.
The ideas underlying topology go back to Gottfried Wilhelm Leibniz, who in the 17th century envisioned the geometria situs and analysis situs. Leonhard Euler's Seven Bridges of Königsberg problem and polyhedron formula are arguably the field's first theorems. The term topology was introduced by Johann Benedict Listing in the 19th century, although, it was not until the first decades of the 20th century that the idea of a topological space was developed.

Motivation
The motivating insight behind topology is that some geometric problems depend not on the exact shape of the objects involved, but rather on the way they are put together. For example, the square and the circle have many properties in common: they are both one-dimensional objects (from a topological point of view) and both separate the plane into two parts, the part inside and the part outside.
In one of the first papers in topology, Leonhard Euler demonstrated that it was impossible to find a route through the town of Königsberg (now Kaliningrad) that would cross each of its seven bridges exactly once. This result did not depend on the lengths of the bridges or on their distance from one another, but only on connectivity properties: which bridges connect to which islands or riverbanks. This Seven Bridges of Königsberg problem led to the branch of mathematics known as graph theory.
Similarly, the hairy ball theorem of algebraic topology says that "one cannot comb the hair flat on a hairy ball without creating a cowlick." This fact is immediately convincing to most people, even though they might not recognize the more formal statement of the theorem, that there is no nonvanishing continuous tangent vector field on the sphere. As with the Bridges of Königsberg, the result does not depend on the shape of the sphere; it applies to any kind of smooth blob, as long as it has no holes.
To deal with these problems that do not rely on the exact shape of the objects, one must be clear about just what properties these problems do rely on. From this need arises the notion of homeomorphism. The impossibility of crossing each bridge just once applies to any arrangement of bridges homeomorphic to those in Königsberg, and the hairy ball theorem applies to any space homeomorphic to a sphere.

Intuitively, two spaces are homeomorphic if one can be deformed into the other without cutting or gluing. A famous example, known as the "Topologist's Breakfast", is that a topologist cannot distinguish a coffee mug from a doughnut. A pliable torus (shaped like a doughnut) can be reshaped to a coffee mug by creating a dimple and progressively enlarging it while shrinking the central hole into the mug's handle.
Homeomorphism can be considered the most basic topological equivalence. Another is homotopy equivalence. This is harder to describe without getting technical, but the essential notion is that two objects are homotopy equivalent if they both result from "squishing" some larger object.

History
Topology, as a well-defined mathematical discipline, originates in the early part of the twentieth century, but some isolated results can be traced back several centuries. Among these are certain questions in geometry investigated by Leonhard Euler. His 1736 paper on the Seven Bridges of Königsberg is regarded as one of the first practical applications of topology. On 14 November 1750, Euler wrote to a friend that he had realized the importance of the edges of a polyhedron. This led to his polyhedron formula, V − E + F = 2 (where V, E, and F respectively indicate the number of vertices, edges, and faces of the polyhedron). Some authorities regard this analysis as the first theorem, signaling the birth of topology.
Further contributions were made by Augustin-Louis Cauchy, Ludwig Schläfli, Johann Benedict Listing, Bernhard Riemann and Enrico Betti. Listing introduced the term "Topologie" in Vorstudien zur Topologie, written in his native German, in 1847, having used the word for ten years in correspondence before its first appearance in print. The English form "topology" was used in 1883 in Listing's obituary in the journal Nature to distinguish "qualitative geometry from the ordinary geometry in which quantitative relations chiefly are treated".
Their work was corrected, consolidated and greatly extended by Henri Poincaré. In 1895, he published his ground-breaking paper on Analysis Situs, which introduced the concepts now known as homotopy and homology, which are now considered part of algebraic topology.

The development of topology in the 20th century was marked by significant advances in both foundational theory and its application to other fields of mathematics. Unifying the work on function spaces of Georg Cantor, Vito Volterra, Cesare Arzelà, Jacques Hadamard, Giulio Ascoli and others, Maurice Fréchet introduced the metric space in 1906. A metric space is now considered a special case of a general topological space, with any given topological space potentially giving rise to many distinct metric spaces. In 1914, Felix Hausdorff coined the term "topological space" and defined what is now called a Hausdorff space. Currently, a topological space is a slight generalization of Hausdorff spaces, given in 1922 by Kazimierz Kuratowski.
Modern topology depends strongly on the ideas of set theory, developed by Georg Cantor in the later part of the 19th century. In addition to establishing the basic ideas of set theory, Cantor considered point sets in Euclidean space as part of his study of Fourier series. For further developments, see point-set topology and algebraic topology.
The 2022 Abel Prize was awarded to Dennis Sullivan "for his groundbreaking contributions to topology in its broadest sense, and in particular its algebraic, geometric and dynamical aspects".

Concepts
Topologies on sets
The term "topology" also refers to a specific mathematical idea central to the area of mathematics called topology. Informally, a topology describes how elements of a set relate spatially to each other. The same set can have different topologies. For instance, the real line, the complex plane, and the Cantor set can be thought of as the same set with different topologies.
Formally, let X be a set and let τ be a family of subsets of X. Then τ is called a topology on X if:

Both the empty set and X are elements of τ.
Any union of elements of τ is an element of τ.
Any intersection of finitely many elements of τ is an element of τ.
If τ is a topology on X, then the pair (X, τ) is called a topological space. The notation Xτ may be used to denote a set X endowed with the particular topology τ. By definition, every topology is a π-system.
The members of τ are called open sets in X. A subset of X is said to be closed if its complement is in τ (that is, its complement is open). A subset of X may be open, closed, both (a clopen set), or neither. The empty set and X itself are always both closed and open. An open subset of X which contains a point x is called an open neighborhood of x.

Continuous functions and homeomorphisms
A function or map from one topological space to another is called continuous if the inverse image of any open set is open. If the function maps the real numbers to the real numbers (both spaces with the standard topology), then this definition of continuous is equivalent to the definition of continuous in calculus. If a continuous function is one-to-one and onto, and if the inverse of the function is also continuous, then the function is called a homeomorphism and the domain of the function is said to be homeomorphic to the range. Another way of saying this is that the function has a natural extension to the topology. If two spaces are homeomorphic, they have identical topological properties and are considered topologically the same. The cube and the sphere are homeomorphic, as are the coffee cup and the doughnut. However, the sphere is not homeomorphic to the doughnut.

Manifolds
While topological spaces can be extremely varied and exotic, many areas of topology focus on the more familiar class of spaces known as manifolds. A manifold is a topological space that resembles Euclidean space near each point. More precisely, each point of an n-dimensional manifold has a neighborhood that is homeomorphic to the Euclidean space of dimension n. Lines and circles, but not figure eights, are one-dimensional manifolds. Two-dimensional manifolds are also called surfaces. Examples include the plane, the sphere, and the torus, which can all be realized  in three dimensions without self-intersection, and the Klein bottle and real projective plane, which cannot.

Subfields
General topology
General topology is the branch of topology dealing with the basic set-theoretic definitions and constructions used in topology. It is the foundation of most other branches of topology, including differential topology, geometric topology, and algebraic topology. Another name for general topology is point-set topology.
The basic object of study is topological spaces, which are sets equipped with a topology, that is, a family of subsets, called open sets, which is closed under finite intersections and (finite or infinite) unions. The fundamental concepts of topology, such as continuity, compactness, and connectedness, can be defined in terms of open sets. Intuitively, continuous functions take nearby points to nearby points. Compact sets are those that can be covered by finitely many sets of arbitrarily small size. Connected sets are sets that cannot be divided into two pieces that are far apart. The words nearby, arbitrarily small, and far apart can all be made precise by using open sets. Several topologies can be defined on a given set. Changing a topology consists of changing the collection of open sets. This changes which functions are continuous and which subsets are compact or connected.
Metric spaces are an important class of topological spaces where the distance between any two points is defined by a function called a metric. In a metric space, an open set is a union of open disks, where an open disk of radius r centered at x is the set of all points whose distance to x is less than r. Many common spaces are topological spaces whose topology can be defined by a metric. This is the case of the real line, the complex plane, real and complex normed vector spaces and Euclidean spaces. Having a metric simplifies many proofs.

Algebraic topology
Algebraic topology is a branch of mathematics that uses tools from algebra to study topological spaces. The basic goal is to find algebraic invariants that classify topological spaces up to homeomorphism, or more commonly up to homotopy equivalence.
The most important of these invariants are homotopy groups, homology, and cohomology.
Although algebraic topology primarily uses algebra to study topological problems, using topology to solve algebraic problems is sometimes also possible. Algebraic topology, for example, allows for a convenient proof that any subgroup of a free group is again a free group.

Differential topology
Differential topology is the field dealing with differentiable functions on differentiable manifolds. It is closely related to differential geometry and together they make up the geometric theory of differentiable manifolds.
More specifically, differential topology considers the properties and structures that require only a smooth structure on a manifold to be defined. Smooth manifolds are "softer" than manifolds with extra geometric structures, which can act as obstructions to certain types of equivalences and deformations that exist in differential topology. For instance, volume and Riemannian curvature are invariants that can distinguish different geometric structures on the same smooth manifold – that is, one can smoothly "flatten out" certain manifolds, but it might require distorting the space and affecting the curvature or volume.

Geometric topology
Geometric topology is a branch of topology that primarily focuses on low-dimensional manifolds (that is, spaces of dimensions 2, 3, and 4) and their interaction with geometry, but it also includes some higher-dimensional topology. Some examples of topics in geometric topology are orientability, handle decompositions, local flatness, crumpling and the planar and higher-dimensional Schönflies theorem.
In high-dimensional topology, characteristic classes are a basic invariant, and surgery theory is a key theory.
Low-dimensional topology is strongly geometric, as reflected in the uniformization theorem in 2 dimensions – every surface admits a constant curvature metric; geometrically, it has one of 3 possible geometries: positive curvature/spherical, zero curvature/flat, and negative curvature/hyperbolic – and the geometrization conjecture (now theorem) in 3 dimensions – every 3-manifold can be cut into pieces, each of which has one of eight possible geometries.
2-dimensional topology can be studied as complex geometry in one variable (Riemann surfaces are complex curves) – by the uniformization theorem every conformal class of metrics is equivalent to a unique complex one, and 4-dimensional topology can be studied from the point of view of complex geometry in two variables (complex surfaces), though not every 4-manifold admits a complex structure.

Generalizations
Occasionally, one needs to use the tools of topology but a "set of points" is not available. In pointless topology one considers instead the lattice of open sets as the basic notion of the theory, while Grothendieck topologies are structures defined on arbitrary categories that allow the definition of sheaves on those categories and with that the definition of general cohomology theories.

Applications
Biology
Topology has been used to study various biological systems including molecules and nanostructure (e.g., membraneous objects). In particular, circuit topology and knot theory have been extensively applied to classify and compare the topology of folded proteins and nucleic acids. Circuit topology classifies folded molecular chains based on the pairwise arrangement of their intra-chain contacts and chain crossings. Knot theory, a branch of topology, is used in biology to study the effects of certain enzymes on DNA. These enzymes cut, twist, and reconnect the DNA, causing knotting with observable effects such as slower electrophoresis.

Computer science
Topological data analysis uses techniques from algebraic topology to determine the large-scale structure of a set (for instance, determining if a cloud of points is spherical or toroidal). The main method used by topological data analysis is to:

Replace a set of data points with a family of simplicial complexes, indexed by a proximity parameter.
Analyse these topological complexes via algebraic topology – specifically, via the theory of persistent homology.
Encode the persistent homology of a data set in the form of a parameterized version of a Betti number, which is called a barcode.
Several branches of programming language semantics, such as domain theory, are formalized using topology. In this context, Steve Vickers, building on work by Samson Abramsky and Michael B. Smyth, characterizes topological spaces as Boolean or Heyting algebras over open sets, which are characterized as semidecidable (equivalently, finitely observable) properties.

Physics
Topology is relevant to physics in areas such as condensed matter physics, quantum field theory, quantum computing and physical cosmology.
The topological dependence of mechanical properties in solids is of interest in the disciplines of mechanical engineering and materials science. Electrical and mechanical properties depend on the arrangement and network structures of molecules and elementary units in materials. The compressive strength of crumpled topologies is studied in attempts to understand the high strength to weight of such structures that are mostly empty space. Topology is of further significance in Contact mechanics where the dependence of stiffness and friction on the dimensionality of surface structures is the subject of interest with applications in multi-body physics.
A topological quantum field theory (or topological field theory or TQFT) is a quantum field theory that computes topological invariants. Although TQFTs were invented by physicists, they are also of mathematical interest, being related to, among other things, knot theory, the theory of four-manifolds in algebraic topology, and the theory of moduli spaces in algebraic geometry. Donaldson, Jones, Witten, and Kontsevich have all won Fields Medals for work related to topological field theory.
The topological classification of Calabi–Yau manifolds has important implications in string theory, as different manifolds can sustain different kinds of strings.
In topological quantum computers, the qubits are stored in topological properties, that are by definition invariant with respect to homotopies.
In cosmology, topology can be used to describe the overall shape of the universe. This area of research is commonly known as spacetime topology.
In condensed matter, a relevant application to topological physics comes from the possibility of obtaining a one-way current, which is a current protected from backscattering. It was first discovered in electronics with the famous quantum Hall effect, and then generalized in other areas of physics, for instance in photonics. David Thouless, Duncan Haldane, and Michael Kosterlitz were awarded the 2016 Nobel Prize in Physics for their work on Topological orders.

Robotics
The possible positions of a robot can be described by a manifold called configuration space. In the area of motion planning, one finds paths between two points in configuration space. These paths represent a motion of the robot's joints and other parts into the desired pose.

Games and puzzles
Disentanglement puzzles are based on topological aspects of the puzzle's shapes and components.

Fiber art
In order to create a continuous join of pieces in a modular construction, it is necessary to create an unbroken path in an order that surrounds each piece and traverses each edge only once. This process is an application of the Eulerian path.

Resources and research
Major journals
Geometry & Topology- a mathematic research journal focused on geometry and topology, and their applications, published by Mathematical Sciences Publishers.
Journal of Topology- a scientific journal which publishes papers of high quality and significance in topology, geometry, and adjacent areas of mathematics.

Major books
Munkres, James R. (2000). Topology (2nd ed.). Upper Saddle River, NJ: Prentice Hall. ISBN 978-0-13-181629-9
Willard, Stephen (2016). General topology. Dover books on mathematics. Mineola, N.Y: Dover publications. ISBN 978-0-486-43479-7
Armstrong, M. A. (1983). Basic topology. Undergraduate texts in mathematics. New York: Springer-Verlag. ISBN 978-0-387-90839-7
John Kelley "General Topology" Springer, 1979.

See also
References
Citations
Bibliography
Aleksandrov, P.S. (1969) [1956]. "Chapter XVIII Topology". In Aleksandrov, A.D.; Kolmogorov, A.N.; Lavrent'ev, M.A. (eds.). Mathematics / Its Content, Methods and Meaning (2nd ed.). The M.I.T. Press.
Croom, Fred H. (1989). Principles of Topology. Saunders College Publishing. ISBN 978-0-03-029804-2.
Richeson, D. (2008). Euler's Gem: The Polyhedron Formula and the Birth of Topology. Princeton University Press.

Further reading
Ryszard Engelking, General Topology, Heldermann Verlag, Sigma Series in Pure Mathematics, December 1989, ISBN 3-88538-006-4.
Bourbaki; Elements of Mathematics: General Topology, Addison–Wesley (1966).
Breitenberger, E. (2006). "Johann Benedict Listing". In James, I.M. (ed.). History of Topology. North Holland. ISBN 978-0-444-82375-5.
Kelley, John L. (1975) [1955]. General Topology. Graduate Texts in Mathematics. Vol. 27 (2nd ed.). New York: Springer-Verlag. ISBN 978-0-387-90125-1. OCLC 1365153.
Brown, Ronald (2006). Topology and Groupoids. Booksurge. ISBN 978-1-4196-2722-4. (Provides a well-motivated, geometric account of general topology, and shows the use of groupoids in discussing van Kampen's theorem, covering spaces, and orbit spaces.)
Wacław Sierpiński, General Topology, Dover Publications, 2000, ISBN 0-486-41148-6
Pickover, Clifford A. (2006). The Möbius Strip: Dr. August Möbius's Marvelous Band in Mathematics, Games, Literature, Art, Technology, and Cosmology. Thunder's Mouth Press. ISBN 978-1-56025-826-1. (Provides a popular introduction to topology and geometry)
Gemignani, Michael C. (1990) [1967]. Elementary Topology (2nd ed.). Dover Publications Inc. ISBN 978-0-486-66522-1.

External links

"Topology, general". Encyclopedia of Mathematics. EMS Press. 2001 [1994].
Elementary Topology: A First Course Viro, Ivanov, Netsvetaev, Kharlamov.
The Topological Zoo at The Geometry Center.
Topology Atlas Archived 28 April 2024 at the Wayback Machine
Topology Course Lecture Notes Archived 5 August 2011 at the Wayback Machine Aisling McCluskey and Brian McMaster, Topology Atlas.
Topology Glossary
Moscow 1935: Topology moving towards America, a historical essay by Hassler Whitney.