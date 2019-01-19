# measles
GPU implementation of measles model

* 2<sup>20</sup> <= population <= 2<sup>28</sup> (allows for all of Nigeria)
* variable susceptibility _not_ needed - susceptibility is 0|1
* waning maternal antibodies _not_ needed - timer switches susceptibility from 0 -> 1
* waning immunity - like maternal antibodies, use timer
* vital dynamics, births and non-disease deaths (births by crude birthrate)
* interventions _do not_ depend on gender
* interventions _may_ depend on age
* spatial distribution and aggregation of contagion by spatial location
* HINT like connectivity of geographic regions
* may need strain tracking to differentiate between endemic and imported
* incidence reporting may include ```time step```, ```spatial location```, ```individual's age```, and ```strain``` (if tracked)
* incidence aggregated by time period and spatial location
