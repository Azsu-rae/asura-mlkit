
= Segmentation

iscontinuité et similarité des pixels.

== Les grandes catégories des méthodes de segmentation

- Segmentation en contours (frontières) : discontinuité
- Segmentation en régions : Similarité des pixels
- Segmentation coopérative (discontinuité et similarité des pixels)
- Classification des pixels
- Binarisation/ seuillage

= Segmentation en contours

== Methods Derivatives

=== Approche Gradient

Le *gradient* d'un pixed au point $(x, y)$ d'une image $I$ est:

$ #sym.nabla I(x, y) = (G_x, G_y) = (frac(#sym.partial I(x, y), #sym.partial x), frac(#sym.partial I(x, y), #sym.partial y))  $

La *norme* ou le *module* $G$ d'un gradient peut etre donnee par plusieurs formules:

$ G = sqrt(G_x^2 + G_y^2) $
$ G = max(|G_x|, |G_y|) $
$ G = |G_x| + |G_y| $

L'*orientation* $θ$ d'un gradient $G$:

$ θ = arctan(frac(G_y, G_x)) $

We use the *Central Difference Operator* to numerically approximate the gradient. The formal central derivative formula is:

$ frac(#sym.partial f, #sym.partial x) = frac(f(x+h) + f(x-h), 2h) $

In images with discrete pixels $h = 1$ and we disregard the division by 2 for compulational speed. So the final formula would look more
like:

$ frac(#sym.partial f, #sym.partial x) = f(x+1) + f(x-1) $

We use this to comput $G_x$ and $G_y$ at a given pixel:

$ G_x = frac(#sym.partial I(x, y), #sym.partial x) = (I(x-1, y), I(x, y), I(x+1, y)) dot (-1, 0, 1) $
$ G_y = frac(#sym.partial I(x, y), #sym.partial y) = mat(I(x, y-1); I(x, y); I(x, y+1)) dot mat(-1; 0; 1) $

A *level line* is a line in which the pixel's levels (values) are the closest. So naturally, the most optimal gradient is perpendicular to
that level line.
