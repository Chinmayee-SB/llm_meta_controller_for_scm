# Design System Reference Card
## Supply Chain LLM+MARL Dashboard

**Purpose**: One-page reference for color, typography, spacing, and component styling  
**Usage**: Keep open while coding to maintain design consistency

---

## COLOR PALETTE

### Agent Colors (Primary)
```
Retailer (Stage 1):    #3B82F6 (Blue)     — Order origin, demand facing
Warehouse (Stage 2):   #10B981 (Teal)     — Middle echelon
Distributor (Stage 3): #F59E0B (Amber)    — Upstream amplification
Manufacturer (Stage 4):#8B5CF6 (Purple)   — Source/production
```

### KPI Status Colors
```
Good/Improving:  #10B981 (Green)    bullwhip ≤ 1.5, service ≥ 90%
Warning/Caution: #F59E0B (Amber)    bullwhip 1.5–2.5, service 80–90%
Critical/Poor:   #EF4444 (Red)      bullwhip > 2.5, service < 80%
Neutral/Inactive:#6B7280 (Gray)     default, no signal
```

### Semantic Colors
```
LLM Decision Highlight: rgba(245, 158, 11, 0.1) background
LLM Border:             #F59E0B (amber border)
Disruption Indicator:   #EF4444 (red pulse)
Text Primary:           var(--color-text-primary)   [system default, black/white]
Text Secondary:         var(--color-text-secondary) [system default, muted]
Background Primary:     var(--color-background-primary)   [white/dark]
Background Secondary:   var(--color-background-secondary) [light gray/darker]
Border Tertiary:        var(--color-border-tertiary) [system default, 0.15 alpha]
```

### CSS Usage Example
```css
/* Agent box styling */
.agent-box.retailer {
  border-color: #3B82F6;
  color: var(--color-text-primary);
}

/* KPI card danger state */
.kpi-card.critical {
  background-color: rgba(239, 68, 68, 0.1); /* Red tint */
  color: #EF4444;
}

/* Dark mode automatic via CSS vars */
@media (prefers-color-scheme: dark) {
  /* CSS variables auto-flip; no manual override needed */
}
```

---

## TYPOGRAPHY

### Font Scale
```
h1 (Dashoard title):     22px, weight 500, var(--color-text-primary)
h2 (Section header):     18px, weight 500, var(--color-text-primary)
h3 (Subsection):         16px, weight 500, var(--color-text-primary)
Body (default):          16px, weight 400, var(--color-text-primary)
Label (input labels):    13px, weight 500, var(--color-text-secondary)
Caption (hints):         12px, weight 400, var(--color-text-secondary)
Data (monospace):        14px, weight 500, fontFamily 'monospace', var(--color-text-primary)
Mini (pipeline labels):  10px, weight 400, var(--color-text-secondary)
```

### Font Families
```
Display/Heading:  var(--font-sans)     (Anthropic Sans, clean sans-serif)
Body:             var(--font-sans)     (same, for consistency)
Monospace/Data:   'monospace'          (system monospace, for numbers)
```

### Weight Rules
```
❌ DON'T: weight 600, 700, 800 (too heavy, looks AI-generated)
✅ DO: weight 400 (regular) + 500 (bold) only
```

### Text Styling Rules
```
✅ Sentence case always (Title Case WRONG, even for headers)
✅ Monospace for numbers: <span style="fontFamily: 'monospace'">12.34</span>
✅ Bold (500) for numbers + labels only: <span style="fontWeight: 500">Cost</span>
❌ DON'T: Underline, italic, ALL CAPS (unless emphasis rare)
❌ DON'T: Hardcoded colors (use CSS vars: var(--color-text-primary))
```

---

## SPACING & LAYOUT

### Spacing Scale (in pixels)
```
XS:  4px    (tight internal spacing)
S:   8px    (compact gaps)
M:  12px    (standard gap between elements)
L:  16px    (comfortable gap, outer spacing)
XL: 20px    (spacious)
XXL:24px    (breathing room, top-level sections)
```

### Vertical Rhythm (use rem)
```
Line height:        1.7 (body text)
Margin between:     1rem (roughly 16px)
Section gap:        1.5rem or 2rem
Padding internal:   1rem (16px default)
```

### Grid & Layout
```
Container width:    680px (fixed, Claude.ai widget standard)
Safe horizontal:    40px margin on each side (600px usable)
Column gap:         12px (default flex gap)
Row gap:            12px (default flex gap)
Flex direction:     row (horizontal on desktop) → column (stacked on mobile)
```

### Responsive Breakpoint
```
Mobile first approach:
  < 700px:  Stack vertically, single column
  ≥ 700px:  Horizontal layout, multi-column grids

Example: KPI cards
  < 700px: grid-template-columns: repeat(2, 1fr) [2×2]
  ≥ 700px: grid-template-columns: repeat(4, 1fr) [1×4]
```

### Component Spacing Patterns
```
KPI Card Grid:      gap: 12px
Agent Box Flex:     gap: 12px (between boxes), 8px (between → connectors)
Chart Container:    padding: 12px (internal), gap: 16px (between charts)
LLM Panel:          padding: 12px, gap: 8px (sections)
Control Bar:        gap: 12px (between buttons)
```

---

## BORDERS & CORNERS

### Border Rules
```
Standard:           0.5px solid var(--color-border-tertiary)
Emphasized:         0.5px solid var(--color-border-secondary) [hover state]
Agent Box:          2px solid {agent-color} [exception: thicker for prominence]
Focus Ring:         2px solid var(--color-border-info) [keyboard focus only]
LLM Panel:          1px solid #F59E0B [amber border for emphasis]

❌ DON'T: 1px borders (too light), 3px+ borders (too heavy)
✅ DO: 0.5px for finesse, 2px only when intentional (focus, thick accent)
```

### Border Radius
```
Standard:           var(--border-radius-md) = 8px
Card:               var(--border-radius-lg) = 12px
Pill/Extreme:       var(--border-radius-xl) = 16px (or rx="8" in SVG)

❌ DON'T: Round on single-sided borders (border-left + rounded = visually broken)
✅ DO: Rounded only on full borders (all 4 sides)

SVG default: rx="4" (subtle), rx="8" (emphatic)
```

---

## SHADOWS & EFFECTS

### Shadow Rules
```
❌ NONE: No drop shadows, blur, glow, or decorative effects (flat design)
✅ FOCUS RING ONLY: box-shadow: 0 0 0 2px var(--color-border-info)
❌ DON'T: box-shadow: 0 2px 8px rgba(0,0,0,0.1) [AI-generated look]
```

### Animation Effects (CSS only)
```
ALLOWED:
  ✅ opacity transitions (fade in/out)
  ✅ transform: translateX/Y/scale (smooth motion)
  ✅ @keyframes pulse, slideIn, fadeOut (defined animations)
  ✅ Background color shifts (200–300ms)

FORBIDDEN:
  ❌ filter: blur(), drop-shadow() [causes layout shift + performance issues]
  ❌ box-shadow with blur [glowy effect, looks AI]
  ❌ Gradients (use solid colors only)
  ❌ Text-shadow
  ❌ Multiple transforms in one animation (simplify)
```

---

## BUTTONS & INTERACTIVE ELEMENTS

### Button States
```
Default:      transparent bg, 0.5px border-secondary, no fill
Hover:        var(--color-background-secondary) bg, 0.5px border-secondary
Active:       scale(0.98) on click (feedback), revert on release
Disabled:     opacity 0.6, cursor: not-allowed, no interaction
Focus:        2px solid var(--color-border-info) focus ring (keyboard)
```

### Button Styling Code
```html
<button style={{
  padding: '8px 16px',
  backgroundColor: 'transparent',
  border: '0.5px solid var(--color-border-secondary)',
  borderRadius: 'var(--border-radius-md)',
  cursor: 'pointer',
  fontWeight: 500,
  transition: 'background-color 150ms ease-out'
}}>
  Action
</button>

/* Hover + active via CSS or onMouseEnter/Leave */
```

### Form Inputs
```
Text Input:    36px height, pre-styled (no custom CSS needed), focus ring built-in
Slider:        4px track, 18px thumb, pre-styled (no custom CSS needed)
Disabled:      opacity 0.6, cursor: not-allowed

❌ DON'T: Override pre-styled inputs (they're optimized)
✅ DO: Use bare <input type="range"> and <input type="text"> tags
```

---

## ANIMATION TIMINGS

### Duration Constants
```
Fast/Feedback:     200ms    (button clicks, color shifts)
Standard:          300ms    (most transitions, slide-in panels)
Slow/Ambient:      800ms    (glow effects, idle animations)
Very Slow:       1200ms    (pulses, loops)
Dismiss:         3000ms    (auto-dismiss panels after N seconds)
```

### Easing Functions
```
ease-out:     Decelerate (recommended for entrance animations)
ease-in:      Accelerate (recommended for exit animations)
ease-in-out:  Smooth throughout (recommended for loops)
linear:       Constant speed (use sparingly, feels robotic)
```

### @keyframes Template
```css
@keyframes slideIn {
  from { 
    opacity: 0; 
    transform: translateY(-10px); 
  }
  to { 
    opacity: 1; 
    transform: translateY(0); 
  }
}

/* Usage */
animation: slideIn 0.3s ease-out;
```

### Animation Rules
```
✅ Always use transform (translateX, translateY, scale) NOT margin/padding
❌ transform: translateX(10px) — GOOD (no layout shift)
❌ margin-left: 10px — BAD (triggers layout recalculation)
```

---

## COMPONENT EXAMPLES

### KPI Card HTML
```html
<div style={{
  backgroundColor: 'var(--color-background-secondary)',
  padding: '12px',
  borderRadius: 'var(--border-radius-md)',
  border: 'none'
}}>
  <div style={{
    fontSize: '12px',
    color: 'var(--color-text-secondary)',
    marginBottom: '4px'
  }}>
    Total Cost
  </div>
  <div style={{
    fontSize: '20px',
    fontWeight: 500,
    fontFamily: 'monospace'
  }}>
    {Math.round(state.cost)}
  </div>
</div>
```

### Agent Box HTML
```html
<div style={{
  flex: 1,
  padding: '12px',
  border: `2px solid #3B82F6`, /* Blue for Retailer */
  borderRadius: 'var(--border-radius-md)',
  backgroundColor: 'transparent',
  transition: 'background-color 150ms ease-out'
}}>
  <div style={{
    fontSize: '12px',
    fontWeight: 500,
    color: 'var(--color-text-secondary)'
  }}>
    Retailer
  </div>
  {/* Child elements */}
</div>
```

### LLM Decision Panel HTML
```html
<div style={{
  backgroundColor: 'rgba(245, 158, 11, 0.1)',
  border: '1px solid #F59E0B',
  borderRadius: 'var(--border-radius-md)',
  padding: '12px',
  animation: 'slideIn 0.3s ease-out'
}}>
  <div style={{
    fontSize: '13px',
    fontWeight: 500,
    marginBottom: '8px',
    color: '#F59E0B'
  }}>
    ✨ LLM Meta-Controller Decision
  </div>
  {/* Content */}
</div>
```

### Slider with Live Display
```html
<label style={{
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  fontSize: '13px'
}}>
  Speed:
  <input 
    type="range" 
    min="0.5" 
    max="2" 
    step="0.5" 
    value={speed}
    onChange={(e) => setSpeed(parseFloat(e.target.value))}
    style={{ width: '100px' }}
  />
  <span style={{ fontFamily: 'monospace' }}>
    {speed.toFixed(1)}×
  </span>
</label>
```

---

## ACCESSIBILITY CHECKLIST

### Colors
```
✅ WCAG AA compliance: 4.5:1 text contrast minimum
✅ Color not only signal: Use shape/text + color (e.g., ↗ + red for "increasing bad")
✅ Dark mode: All colors tested in dark mode
❌ DON'T: Rely on color alone (red alone ≠ understandable)
```

### Keyboard Navigation
```
✅ All buttons focusable (tab index visible)
✅ Enter/Space triggers action
✅ Slider drag with arrow keys (native input)
✅ Focus ring visible (2px solid var(--color-border-info))
❌ DON'T: Move focus programmatically (disorienting)
```

### Screen Readers
```
✅ Button aria-label for icon-only buttons
✅ Canvas/SVG has role="img" + <title> + <desc>
✅ Form labels properly associated (label htmlFor)
✅ Semantics: use <button>, <input>, <table> not <div onclick>
```

### Motion Sensitivity
```
✅ Respect @media (prefers-reduced-motion: reduce)
   — Disable animations, instantly show state
❌ DON'T: Force animations on all users
```

---

## COMMON MISTAKES & FIXES

| Mistake | Why Bad | Fix |
|---------|---------|-----|
| Hardcoded colors `color: #333` | Invisible in dark mode | Use `var(--color-text-primary)` |
| `padding: 10px 15px 8px` (asymmetric) | Looks sloppy | Use grid: 8px, 12px, 16px increments |
| `box-shadow: 0 2px 8px rgba(...)` | AI-generated look, perf hit | Use solid borders or remove |
| `margin-left: 20px` in animation | Jank, layout shift | Use `transform: translateX(20px)` |
| `font-weight: 700` | Too heavy vs. system font | Use 400 or 500 only |
| `font-size: 9px` | Unreadable | Min 11px, prefer 12px+ |
| `border-radius: 20px` on small box | Becomes pill, loses identity | Max 8–12px |
| `opacity: 0` hidden element | Screen readers still see | Use `display: none` or `visibility: hidden` |
| Multiple nested animations | Janky, overlapping effects | Simplify: one animation per element |
| `@media (prefers-color-scheme)` with hardcoded colors | Manual dark mode redundant | Use CSS vars only |

---

## RESPONSIVE DESIGN CHECKLIST

### Mobile (< 700px)
```
Layout:
  ☐ KPI cards: 2×2 grid (not 1×4)
  ☐ Agent boxes: Vertical stack (not horizontal flex)
  ☐ Charts: Full width, no side-by-side
  ☐ Control bar: Wrap buttons to 2–3 lines (no overflow)

Touch:
  ☐ All buttons ≥ 44px height
  ☐ Slider thumb ≥ 18px diameter
  ☐ Tap target spacing: ≥ 8px between targets

Text:
  ☐ Labels readable (≥ 12px)
  ☐ Numbers visible (≥ 14px monospace)
  ☐ No text overflow (use line-break if needed)
```

### Desktop (≥ 700px)
```
Layout:
  ☐ KPI cards: 1×4 grid (full width)
  ☐ Agent boxes: Horizontal flex (all visible)
  ☐ Charts: 2-column layout (side-by-side)
  ☐ Control bar: Inline (no wrapping)
```

### Testing
```
✅ Always test at 680px (widget container width)
✅ Use Chrome DevTools responsive mode
✅ Test portrait + landscape on mobile
✅ Pinch-zoom should work (no viewport scaling issues)
```

---

## QUICK COPY-PASTE SNIPPETS

### Box / Card Container
```jsx
<div style={{
  backgroundColor: 'var(--color-background-secondary)',
  border: '0.5px solid var(--color-border-tertiary)',
  borderRadius: 'var(--border-radius-lg)',
  padding: '16px',
  display: 'flex',
  flexDirection: 'column',
  gap: '12px'
}}>
  {/* Content */}
</div>
```

### Flexbox Row (spacing)
```jsx
<div style={{
  display: 'flex',
  gap: '12px',
  alignItems: 'center',
  flexWrap: 'wrap'
}}>
  {/* Items */}
</div>
```

### Grid (4 columns, responsive)
```jsx
<div style={{
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
  gap: '12px'
}}>
  {/* Items */}
</div>
```

### Monospace Data Display
```jsx
<span style={{
  fontFamily: 'monospace',
  fontWeight: 500,
  fontSize: '14px'
}}>
  {Math.round(value)}
</span>
```

### Conditional Color (danger)
```jsx
<div style={{
  color: value > threshold ? '#EF4444' : 'var(--color-text-primary)',
  fontWeight: 500
}}>
  {displayValue}
</div>
```

---

## FINAL CHECKLIST (Before Commit)

```
Typography:
  ☐ All font sizes from scale (10/12/13/14/16/18/22px)
  ☐ Weights: 400 or 500 only (no 600/700)
  ☐ All text uses CSS vars (no hardcoded colors)
  ☐ Sentence case (no Title Case)

Colors:
  ☐ Agent colors correct (#3B82F6 / #10B981 / #F59E0B / #8B5CF6)
  ☐ KPI status colors applied (green/amber/red)
  ☐ CSS vars used (var(--color-text-primary) etc.)
  ☐ Dark mode tested (readable in dark mode)
  ☐ Contrast ≥ 4.5:1

Spacing:
  ☐ Padding/margin from scale (8/12/16/20/24px)
  ☐ Gaps consistent (12px standard)
  ☐ Vertical rhythm (1rem between sections)
  ☐ Mobile breakpoints tested (680px)

Interaction:
  ☐ Animations use transform (not margin)
  ☐ Durations correct (200/300/800ms)
  ☐ Focus rings visible (2px solid info color)
  ☐ Keyboard nav works (Tab through all)

Accessibility:
  ☐ ARIA labels on icon buttons
  ☐ prefers-reduced-motion respected
  ☐ Screen reader tested
  ☐ Color contrast OK

Performance:
  ☐ Components memoized (no unnecessary re-renders)
  ☐ 60 FPS smooth (Chrome DevTools FPS meter)
  ☐ No console errors/warnings
  ☐ Mobile touch responsive (no lag)
```

---

**Keep this card open while coding. Reference often!**