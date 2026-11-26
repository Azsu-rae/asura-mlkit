#set page(width: auto, height: auto, margin: 6pt, fill: none)
#set text(fill: rgb("#cdd6f4"))                      // keep light text if it will sit on a dark page

#import "@preview/cetz:0.5.3"
#cetz.canvas({
  import cetz.draw: *

  let displayed_layers = (16, 14, 10) // displayed sizes
  let layers = (784, 128, 10)         // actual sizes
  let cap = 10                        // max slots drawn per layer
  let gap-x = 8
  let gap-y = 1.2

  let is-dots(l, k) = k == int(displayed_layers.at(l) / 2) and l != displayed_layers.len() - 1
  let pos(l, k) = (l * gap-x, -(k - (displayed_layers.at(l) - 1) / 2) * gap-y)
  let num(l, k) = {
    let half = int(displayed_layers.at(l)/2)
    if k > half {
      layers.at(l) - half + (k - half)
    } else {
      k
    }
  }
  let exp(l, k) = {
    if l == 0 {
      $p_#num(l, k + 1)$
    } else if l < layers.len() - 1 {
      $a^((#l))_#num(l, k + 1)$
    } else {
      $#k$
    }
  }

  // connections, skipping the dots slots
  for l in range(displayed_layers.len() - 1) {
    for i in range(displayed_layers.at(l)) {
      for j in range(displayed_layers.at(l + 1)) {
        if not is-dots(l, i) and not is-dots(l + 1, j) {
          line(pos(l, i), pos(l + 1, j), stroke: 0.4pt + rgb("#6c7086"))
        }
      }
    }
  }

  // neurons and dots
  for l in range(displayed_layers.len()) {
    for k in range(displayed_layers.at(l)) {
      if is-dots(l, k) {
        content(pos(l, k), $dots.v$)
      } else {
        circle(pos(l, k), radius: 0.35, fill: black, stroke: 0.8pt + rgb("#6c7086"))
        content(pos(l, k), text(size: 9pt, fill: white, exp(l, k)))
      }
    }
  }
})
