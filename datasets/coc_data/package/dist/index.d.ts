type Base = 'home' | 'builder' | 'clan_capital';
type Category = 'defense' | 'crafted-defense' | 'wall' | 'trap' | 'troop' | 'spell' | 'resource' | 'army' | 'research' | 'guardian' | 'siege-machine' | 'pet' | 'hero' | 'hero-equipment' | 'town-hall' | 'other';
type ResourceType = 'Gold' | 'Elixir' | 'Dark Elixir' | 'Gold or Elixir' | 'Builder Gold' | 'Builder Elixir' | 'Builder Gold or Builder Elixir' | 'Capital Gold' | 'Gems';
interface BuildTime {
    days: number;
    hours: number;
    minutes: number;
    seconds: number;
}
interface DonationCost {
    amount: number;
    resource: 'Elixir' | 'Dark Elixir';
    gemsOrRaidMedals: number;
}
interface TownHallAvailability {
    townHallLevel: number;
    count: number;
    countAfterMerges?: number;
}
interface BuilderHallAvailability {
    builderHallLevel: number;
    count: number;
}
interface DistrictHallAvailability {
    districtHallLevel: number;
    count: number;
}
interface CapitalHallAvailability {
    capitalHallLevel: number;
    count: number;
}
type ClanCapitalDistrict = 'barbarianCamp' | 'wizardValley' | 'balloonLagoon' | 'buildersWorkshop' | 'dragonCliffs' | 'golemQuarry' | 'skeletonPark' | 'goblinMines';
/** Counts indexed by District Hall level minus 1 (index 0 = DH1, index 1 = DH2, etc.). */
interface DistrictAvailability {
    district: ClanCapitalDistrict;
    countPerDistrictHall: number[];
}
interface BuilderHallLevelCounts {
    /** count × maxLevel for defenses, army buildings, and resource buildings */
    structures: number;
    /** count × maxLevel for traps */
    traps: number;
    /** sum of all troop levels where starLabRequired <= max Star Lab level at bhLevel */
    starLab: number;
    /** sum of all hero levels where builderHallLevelRequired <= bhLevel */
    heroes: number;
    /** wallCount × max wall tier level at bhLevel */
    walls: number;
    /** sum of all five categories */
    total: number;
}
interface ClanCapitalDistrictCounts {
    /** sum of count × maxLevel for all buildings in this district/zone */
    structures: number;
    /** wallCount × max wall level available in this district/zone */
    walls: number;
    /** structures + walls */
    total: number;
}
interface ClanCapitalLevelCounts {
    /** Capital Peak zone: buildings gated by capitalHallLevel, walls gated by capitalHallLevel */
    capitalPeak: ClanCapitalDistrictCounts;
    /** Barbarian Camp district (unlocks at Capital Hall 2) */
    barbarianCamp: ClanCapitalDistrictCounts;
    /** Wizard Valley district (unlocks at Capital Hall 3) */
    wizardValley: ClanCapitalDistrictCounts;
    /** Balloon Lagoon district (unlocks at Capital Hall 4) */
    balloonLagoon: ClanCapitalDistrictCounts;
    /** Builder's Workshop district (unlocks at Capital Hall 5) */
    buildersWorkshop: ClanCapitalDistrictCounts;
    /** Dragon Cliffs district (unlocks at Capital Hall 6) */
    dragonCliffs: ClanCapitalDistrictCounts;
    /** Golem Quarry district (unlocks at Capital Hall 7) */
    golemQuarry: ClanCapitalDistrictCounts;
    /** Skeleton Park district (unlocks at Capital Hall 8) */
    skeletonPark: ClanCapitalDistrictCounts;
    /** Goblin Mines district (unlocks at Capital Hall 9) */
    goblinMines: ClanCapitalDistrictCounts;
    /** sum of max troop levels where districtHallRequired <= max DH level available */
    troops: number;
    /** sum of max spell levels where districtHallRequired <= max DH level available */
    spells: number;
    /** sum of all categories */
    total: number;
}
interface TownHallLevelCounts {
    /** count × maxNormalLevel for defenses, army buildings, and resource buildings */
    structures: number;
    /** count × maxLevel for traps */
    traps: number;
    /** count × #superchargeLevel for defenses and resource buildings */
    superCharge: number;
    /** sum of all troop + spell + siege machine levels where townHallRequired <= thLevel */
    lab: number;
    /** sum of heroLevelCaps from the max Hero Hall level available at thLevel */
    heroes: number;
    /** count × maxLevel for guardians */
    guardians: number;
    /** sum of all hero equipment levels where blacksmithLevelRequired <= max blacksmith level at thLevel */
    equipment: number;
    /** sum of all pet levels where townHallRequired <= thLevel */
    pets: number;
    /** sum of all module upgrade levels across all currently active crafted defenses */
    craftedDefenses: number;
    /** wallCount × max wall tier level at thLevel */
    walls: number;
    /** sum of all ten categories */
    total: number;
}

interface BuildingLevel {
    level: number;
    hitpoints: number;
    buildCost: number;
    buildCostResource: ResourceType;
    buildTime: BuildTime;
    xpGained: number;
}
interface Building<L extends BuildingLevel = BuildingLevel> {
    id: string;
    name: string;
    description?: string;
    base: Base;
    category: Category;
    size: string;
    levels: L[];
}

interface ArmyBuildingLevel extends BuildingLevel {
    housingSpace: number;
    images: {
        normal: string;
    };
}
interface HomeArmyBuildingLevel extends ArmyBuildingLevel {
    townHallRequired: number;
}
interface HomeArmyBuilding extends Building<HomeArmyBuildingLevel> {
    availablePerTownHall: TownHallAvailability[];
}
interface BuilderArmyBuildingLevel extends ArmyBuildingLevel {
    builderHallRequired: number;
}
interface BuilderArmyBuilding extends Building<BuilderArmyBuildingLevel> {
    availablePerBuilderHall: BuilderHallAvailability[];
}
interface BuilderArmyCampLevel {
    level: number;
    hitpoints: number;
    housingSpace: number;
    images: {
        normal: string;
    };
}
interface BuilderArmyCampInstance {
    instance: number;
    buildCost: number;
    buildCostResource: ResourceType;
    buildTime: BuildTime;
    xpGained: number;
    builderHallRequired: number;
}
interface BuilderArmyCampBuilding {
    id: string;
    name: string;
    description?: string;
    base: Base;
    category: Category;
    size: string;
    levels: BuilderArmyCampLevel[];
    instances: BuilderArmyCampInstance[];
    availablePerBuilderHall: BuilderHallAvailability[];
}
interface BuilderHealingHutBuildingLevel extends BuildingLevel {
    healthRecovery: number;
    builderHallRequired: number;
    images: {
        normal: string;
    };
}
interface BuilderHealingHutBuilding extends Building<BuilderHealingHutBuildingLevel> {
    availablePerBuilderHall: BuilderHallAvailability[];
}
interface BuilderBarracksBuildingLevel extends BarracksBuildingLevel {
    builderHallRequired: number;
}
interface BuilderBarracksBuilding extends Building<BuilderBarracksBuildingLevel> {
    availablePerBuilderHall: BuilderHallAvailability[];
}
interface BarracksBuildingLevel extends BuildingLevel {
    unlockedUnit: string;
    images: {
        normal: string;
    };
}
interface HomeBarracksBuildingLevel extends BarracksBuildingLevel {
    townHallRequired: number;
}
interface HomeBarracksBuilding extends Building<HomeBarracksBuildingLevel> {
    availablePerTownHall: TownHallAvailability[];
}
interface SpellFactoryBuildingLevel extends BuildingLevel {
    unlockedSpell: string | string[];
    spellStorageCapacity: number;
    images: {
        normal: string;
    };
}
interface HomeSpellFactoryBuildingLevel extends SpellFactoryBuildingLevel {
    townHallRequired: number;
}
interface HomeSpellFactoryBuilding extends Building<HomeSpellFactoryBuildingLevel> {
    availablePerTownHall: TownHallAvailability[];
}
interface ClanCastleLevel extends BuildingLevel {
    troopCapacity: number;
    spellCapacity: number;
    siegeMachineCapacity: number;
    labLevelCap: number;
    images: {
        normal: string;
    };
}
interface HomeClanCastleLevel extends ClanCastleLevel {
    townHallRequired: number;
}
interface HomeClanCastle extends Building<HomeClanCastleLevel> {
    triggerRadius: number;
    availablePerTownHall: TownHallAvailability[];
}
interface BlacksmithOreCapacity {
    shinyOre: number;
    glowyOre: number;
    starryOre: number;
}
interface BlacksmithMaxEquipmentLevel {
    common: number;
    epic: number;
}
interface HomeBlacksmithBuildingLevel extends BuildingLevel {
    equipmentUnlocked: string | string[] | null;
    oreCapacity: BlacksmithOreCapacity;
    maxEquipmentLevel: BlacksmithMaxEquipmentLevel;
    townHallRequired: number;
    images: {
        normal: string;
    };
}
interface HomeBlacksmithBuilding extends Building<HomeBlacksmithBuildingLevel> {
    availablePerTownHall: TownHallAvailability[];
}
interface HeroLevelCaps {
    barbarianKing?: number;
    archerQueen?: number;
    minionPrince?: number;
    grandWarden?: number;
    royalChampion?: number;
    dragonDuke?: number;
}
interface HomeHeroHallBuildingLevel extends BuildingLevel {
    unlockedHero: string | null;
    heroSlots: number;
    heroLevelCaps: HeroLevelCaps;
    townHallRequired: number;
    images: {
        normal: string;
        active: string;
    };
}
interface HomeHeroHallBuilding extends Building<HomeHeroHallBuildingLevel> {
    availablePerTownHall: TownHallAvailability[];
}
interface HeroBannerImages {
    empty: string;
    barbarianKing: string;
    archerQueen: string;
    minionPrince: string;
    grandWarden: string;
    royalChampion: string;
    dragonDuke: string;
}
interface HeroBannerBuilding {
    id: string;
    name: string;
    description?: string;
    base: Base;
    category: Category;
    size: string;
    images: HeroBannerImages;
    availablePerTownHall: TownHallAvailability[];
}
interface HomeWorkshopBuildingLevel extends BuildingLevel {
    unlockedSiegeMachine: string;
    siegeMachineCapacity: number;
    townHallRequired: number;
    images: {
        normal: string;
    };
}
interface HomeWorkshopBuilding extends Building<HomeWorkshopBuildingLevel> {
    availablePerTownHall: TownHallAvailability[];
}
interface HomePetHouseBuildingLevel extends BuildingLevel {
    unlockedPet: string;
    townHallRequired: number;
    images: {
        normal: string;
    };
}
interface HomePetHouseBuilding extends Building<HomePetHouseBuildingLevel> {
    availablePerTownHall: TownHallAvailability[];
}
interface ClanCapitalArmyBuildingLevel extends BuildingLevel {
    housingSpace: number;
    districtHallRequired: number;
    images: {
        normal: string;
    };
}
interface ClanCapitalArmyBuilding extends Building<ClanCapitalArmyBuildingLevel> {
    availablePerDistrict: DistrictAvailability[];
}
interface ClanCapitalSpellStorageBuildingLevel extends BuildingLevel {
    spellCapacity: number;
    districtHallRequired: number;
    images: {
        normal: string;
    };
}
interface ClanCapitalSpellStorageBuilding extends Building<ClanCapitalSpellStorageBuildingLevel> {
    availablePerDistrict: DistrictAvailability[];
}
interface ClanCapitalBarracksBuildingLevel extends BuildingLevel {
    districtHallRequired: number;
    images: {
        normal: string;
    };
}
interface ClanCapitalBarracksBuilding extends Building<ClanCapitalBarracksBuildingLevel> {
    troopUnlocked: string;
    availablePerDistrict: DistrictAvailability[];
}
interface ClanCapitalSpellFactoryLevel extends BuildingLevel {
    districtHallRequired: number;
    images: {
        normal: string;
    };
}
interface ClanCapitalSpellFactory extends Building<ClanCapitalSpellFactoryLevel> {
    spellUnlocked: string;
    availablePerDistrict: DistrictAvailability[];
}

interface CapitalHallWeaponMode {
    range: number;
    attackSpeed: number;
    damageType: string;
    targetType: 'ground' | 'air' | 'both';
}
interface CapitalHallLevel {
    level: number;
    hitpoints: number;
    buildCost: number;
    buildCostResource: ResourceType;
    buildTime: BuildTime;
    xpGained: number;
    capitalUpgradesRequired?: number;
    newCapitalUpgradesRequired?: number;
    maxBuildings: number;
    dps?: number;
    damagePerShot?: number;
    images: {
        normal: string;
        active?: string;
    };
}
interface CapitalHall {
    id: string;
    name: string;
    base: 'clan_capital';
    category: 'capital-hall';
    size: string;
    description?: string;
    weaponMode: CapitalHallWeaponMode;
    levels: CapitalHallLevel[];
}

interface DistrictHallCapitalHallRequired {
    barbarianCamp: number;
    wizardValley: number;
    balloonLagoon: number;
    buildersWorkshop: number;
    dragonCliffs: number;
    golemQuarry: number;
    skeletonPark?: number;
    goblinMines?: number;
}
interface DistrictHallLevel {
    level: number;
    hitpoints: number;
    buildCost: number;
    buildCostResource: ResourceType;
    buildTime: BuildTime;
    xpGained: number;
    capitalHallRequired: DistrictHallCapitalHallRequired;
    images: {
        normal: string;
    };
}
interface DistrictHall {
    id: string;
    name: string;
    base: 'clan_capital';
    category: 'district-hall';
    size: string;
    description?: string;
    levels: DistrictHallLevel[];
}

interface BuilderHallLevel {
    level: number;
    hitpoints: number;
    buildCost: number;
    buildCostResource: ResourceType;
    buildTime: BuildTime;
    xpGained: number;
    maxBuildings: number;
    images: {
        normal: string;
    };
}
interface BuilderHall {
    id: string;
    name: string;
    base: 'builder';
    category: 'builder-hall';
    size: string;
    description?: string;
    levels: BuilderHallLevel[];
}

type EquipmentRarity = 'Common' | 'Rare' | 'Epic';
interface HeroEquipmentLevel {
    level: number;
    hitpointIncrease?: number;
    hpRecoveryIncrease?: number;
    upgradeShinyOre: number;
    upgradeGlowingOre: number;
    upgradeStarryOre: number;
    blacksmithLevelRequired: number;
    stats: Record<string, string | number>;
}
interface HeroEquipment {
    id: string;
    name: string;
    description?: string;
    base: 'home';
    category: 'hero-equipment';
    hero: string;
    rarity: EquipmentRarity;
    abilityType: 'Active' | 'Passive';
    unlockRequirement: string[];
    ability?: Record<string, string | number>;
    images: {
        icon: string;
    };
    levels: HeroEquipmentLevel[];
}

interface BuilderHeroAbilityChargeLevel {
    attackType: string;
    /** For Chain Lightning: damage decay between targets */
    chainDamageDecay?: number;
    /** For continuous-attack modes: damage per second */
    dps?: number;
    /** For continuous-attack modes: damage per hit */
    dph?: number;
    /** For bomb-drop modes: total damage of the bomb */
    damage?: number;
    /** For splash/bomb modes: blast radius in tiles */
    damageRadius?: number;
    healthRecovery: number;
    targets?: number;
}
interface BuilderHeroAbilityLevel {
    abilityLevel: number;
    chargeLevel1: BuilderHeroAbilityChargeLevel;
    chargeLevel2: BuilderHeroAbilityChargeLevel;
    chargeLevel3: BuilderHeroAbilityChargeLevel;
}
interface BuilderHeroAbility {
    name: string;
    cooldownPerChargeLevel: number;
    /** Ability ends after this many hits (Electric Hammer). */
    durationHits?: number;
    /** Movement speed while ability is active (Bomb Rush). */
    movementSpeedDuringAbility?: number;
    levels: BuilderHeroAbilityLevel[];
}
interface BuilderHeroLevel {
    level: number;
    damagePerSecond: number;
    damagePerHit: number;
    hitpoints: number;
    abilityLevel: number | null;
    builderHallLevelRequired: number;
    upgradeCost: number;
    upgradeCostResource: 'Builder Elixir';
    upgradeTime: BuildTime;
}
interface BuilderHero {
    id: string;
    name: string;
    description?: string;
    base: 'builder';
    category: 'hero';
    preferredTarget: string;
    attackType: string;
    movementSpeed: number;
    attackSpeed: number;
    range: number;
    specialAbility?: string;
    ability?: BuilderHeroAbility;
    images: {
        icon: string;
    };
    levels: BuilderHeroLevel[];
}
interface HomeHeroLevel {
    level: number;
    damagePerSecond: number;
    damagePerHit: number;
    hitpoints: number;
    healthRecovery: number;
    heroHallLevelRequired: number;
    upgradeCost: number;
    upgradeCostResource: ResourceType;
    upgradeTime: BuildTime;
}
interface HomeHero {
    id: string;
    name: string;
    description?: string;
    base: 'home';
    category: 'hero';
    preferredTarget: string;
    attackType: string;
    movementSpeed: number;
    attackSpeed: number;
    range: number;
    searchRadius: number;
    specialAbility?: string;
    images: {
        icon: string;
    };
    levels: HomeHeroLevel[];
}

interface ResearchBuildingLevel extends BuildingLevel {
    boostCost?: number;
    images: {
        normal: string;
    };
}
interface HomeResearchBuildingLevel extends ResearchBuildingLevel {
    townHallRequired: number;
}
interface HomeResearchBuilding extends Building<HomeResearchBuildingLevel> {
    availablePerTownHall: TownHallAvailability[];
}
interface BuilderResearchBuildingLevel extends ResearchBuildingLevel {
    builderHallRequired: number;
}
type BuilderResearchBuilding = Building<BuilderResearchBuildingLevel>;

interface TownHallStorageCapacity {
    gold: number;
    elixir: number;
    darkElixir: number;
}
interface TownHallWeaponLevel {
    level: number;
    dps: number;
    damagePerHit: number;
    buildCost: number;
    buildCostResource: ResourceType;
    buildTime: BuildTime;
    xpGained: number;
    images: {
        normal: string;
        townHall?: string;
    };
}
interface TownHallWeapon {
    name: string;
    hitpoints: number;
    targets: number;
    range: number;
    attackSpeed: number;
    targetType: 'ground' | 'air' | 'both';
    damageType: string;
    deathDamage?: number;
    deathDamageRadius?: number;
    deathSpeedDecrease?: number;
    deathAttackRateDecrease?: number;
    deathPoisonMaxDps?: number;
    deathPoisonDuration?: number;
    flameMaxDps?: number;
    flameDuration?: number;
    levels: TownHallWeaponLevel[];
}
interface TownHallLevel {
    level: number;
    hitpoints: number;
    buildCost: number;
    buildCostResource: ResourceType;
    buildTime: BuildTime;
    xpGained: number;
    maxBuildings: number;
    maxTraps: number;
    storageCapacity: TownHallStorageCapacity;
    weapon: TownHallWeapon | null;
    images: {
        normal: string;
    };
}
interface TownHall {
    id: string;
    name: string;
    base: 'home';
    category: 'town-hall';
    size: string;
    description?: string;
    levels: TownHallLevel[];
}

interface CraftedDefenseModuleUpgrade {
    level: number;
    stat: number;
    buildCost: number;
    buildCostResource: ResourceType;
    buildTime: BuildTime;
    xpGained: number;
    sparkyStones: number;
}
interface CraftedDefenseModule {
    name: string;
    controls: string;
    upgrades: CraftedDefenseModuleUpgrade[];
}
interface CraftedDefenseImageEntry {
    fromEffectiveLevel: number;
    toEffectiveLevel: number;
    normal: string;
}
interface CraftedDefense {
    id: string;
    name: string;
    description?: string;
    base: 'home';
    category: 'crafted-defense';
    size: string;
    craftingPhase: number;
    isCurrent: boolean;
    targetType: 'ground' | 'air' | 'both';
    modules: [CraftedDefenseModule, CraftedDefenseModule, CraftedDefenseModule];
    images: CraftedDefenseImageEntry[];
}

interface BuilderStats {
    repairPerSecond: number;
    repairPerHit: number;
}
interface BuilderMode {
    range: number;
    repairSpeed: number;
    movementSpeed: number;
}
interface DefenseModeStats {
    dps?: number;
    damagePerShot?: number;
    bonusDamagePercent?: number;
    secondaryChainDamagePerShot?: number;
    tertiaryChainDamagePerShot?: number;
    pushStrength?: number;
    dpsInitial?: number;
    dpsAfter1p5s?: number;
    dpsAfter2p5s?: number;
    dpsAfter7p5s?: number;
    numberOfTargets?: number;
    shockwaveDamagePerHit?: number;
    damagePerShotMin?: number;
    splashDamageMax?: number;
    splashDamageMin?: number;
    shotsPerBurst?: number;
    timeBetweenBursts?: number;
    burnDps?: number;
    totalBurnDamage?: number;
    burnDamagePerTick?: number;
}
interface DefenseMode {
    range: number;
    minRange?: number;
    attackSpeed: number;
    damageType: 'single' | 'splash' | 'none' | 'multiple' | 'ricochet' | 'chain';
    maxChainTargets?: number;
    chainRange?: number;
    numberOfTargets?: number;
    splashRadius?: number;
    triggerRange?: number;
    shotsPerBurst?: number;
    timeBetweenBursts?: number;
    activationHousingSpace?: number;
    numberOfRounds?: number;
    lavaLifetime?: number;
    lavaRadius?: number;
    burnDamageTickRate?: number;
}
interface BurstDefenseMode extends DefenseMode {
    shotsPerBurst: number;
    timeBetweenBursts: number;
    availableFromLevel: number;
}
interface GearUp {
    cost: number;
    costResource: ResourceType;
    time: BuildTime;
    requiresLevel: number;
    requiresBuilderBuilding?: string;
    requiresBuilderBuildingLevel?: number;
}
interface HomeDefenseLevel extends BuildingLevel {
    townHallRequired: number;
    supercharge?: boolean;
    deathDamage?: number;
    unlocksSpell?: string;
    stats: {
        normal: DefenseModeStats;
        gearedUpBurst?: DefenseModeStats;
        gearedUpFastAttack?: DefenseModeStats;
        multiTarget?: DefenseModeStats;
        fastAttack?: DefenseModeStats;
        stage1?: DefenseModeStats;
        stage2?: DefenseModeStats;
        stage3?: DefenseModeStats;
        builder?: BuilderStats;
    };
    images: {
        normal: string;
        gearedUpNormal?: string;
        gearedUpBurst?: string;
        gearedUpFastAttack?: string;
        airAndGround?: string;
        groundDepleted?: string;
        airAndGroundDepleted?: string;
        multiTarget?: string;
        fastAttack?: string;
        dormant?: string;
        stage1?: string;
        stage2?: string;
        stage3?: string;
        singleTargetDepleted?: string;
        multiTargetDepleted?: string;
        headDown?: string;
        unloaded?: string;
        depleted?: string;
        active?: string;
        poison?: string;
        invisibility?: string;
        earthquake?: string;
    };
}
interface SpellTowerMode {
    range: number;
    spellRadius: number;
    rechargeTime: number;
}
interface RageSpellMode extends SpellTowerMode {
    spellDuration: number;
    damageIncrease: number;
}
interface PoisonSpellMode extends SpellTowerMode {
    spellDuration: number;
    maxDps: number;
    speedDecrease: number;
    attackRateDecrease: number;
}
interface InvisibilitySpellMode extends SpellTowerMode {
    spellDuration: number;
}
interface EarthquakeSpellMode extends SpellTowerMode {
    troopDamagePercent: number;
}
interface HomeDefense extends Building<HomeDefenseLevel> {
    targetType: 'ground' | 'air' | 'both';
    modes: {
        normal?: DefenseMode;
        gearedUpBurst?: BurstDefenseMode;
        gearedUpFastAttack?: DefenseMode;
        airAndGround?: DefenseMode;
        multiTarget?: DefenseMode;
        fastAttack?: BurstDefenseMode;
        stage1?: DefenseMode;
        stage2?: DefenseMode;
        stage3?: DefenseMode;
        builder?: BuilderMode;
        rage?: RageSpellMode;
        poison?: PoisonSpellMode;
        invisibility?: InvisibilitySpellMode;
        earthquake?: EarthquakeSpellMode;
    };
    gearUp?: GearUp;
    specialAbility?: string;
    availablePerTownHall: TownHallAvailability[];
    placementCosts?: Array<{
        instance: number;
        cost: number;
        costResource: ResourceType;
    }>;
}
interface BuilderDefenseLevel extends BuildingLevel {
    builderHallRequired: number;
    troopLevel?: number;
    spawnCount?: number;
    stats: {
        normal: DefenseModeStats;
        fastAttack?: DefenseModeStats;
    };
    images: {
        normal: string;
    };
}
interface BuilderDefense extends Building<BuilderDefenseLevel> {
    targetType: 'ground' | 'air' | 'both';
    modes: {
        normal: DefenseMode;
        fastAttack?: DefenseMode;
    };
    defendingTroops?: Array<{
        name: string;
        count: number;
    }>;
    specialAbility?: string;
    availablePerBuilderHall: BuilderHallAvailability[];
}
interface ClanCapitalDefenseLevel extends BuildingLevel {
    capitalHallRequired?: number;
    districtHallRequired: number;
    deathDamage?: number;
    troopLevel?: number;
    stats: {
        normal: DefenseModeStats;
    };
    images: {
        normal: string;
    };
}
interface ClanCapitalDefense extends Building<ClanCapitalDefenseLevel> {
    targetType: 'ground' | 'air' | 'both';
    modes: {
        normal: DefenseMode;
    };
    defendingTroops?: Array<{
        name: string;
        count: number;
    }>;
    availablePerCapitalHall?: CapitalHallAvailability[];
    availablePerDistrict: DistrictAvailability[];
}

interface GuardianMode {
    attackSpeed: number;
    damageType: 'splash' | 'single';
    range: number;
    damageRadius?: number;
    deathDamageRadius?: number;
    movementSpeed: number;
    patrolRadius?: number;
    searchRadius?: number;
    triggerRadius: number;
}
interface GuardianLevel extends BuildingLevel {
    townHallRequired: number;
    deathDamage?: number;
    stats: {
        normal: DefenseModeStats;
    };
    images: {
        normal: string;
    };
}
interface Guardian extends Building<GuardianLevel> {
    guardianType: string;
    targetType: 'ground' | 'air' | 'both';
    mode: GuardianMode;
    specialAbility?: string;
    availablePerTownHall: TownHallAvailability[];
}
interface LongshotGuardian extends Guardian {
    guardianType: 'longshot';
}
interface SmasherGuardian extends Guardian {
    guardianType: 'smasher';
    rageSpeedIncrease: number;
    rageDamageIncrease: number;
}

interface OtherBuildingLevel extends BuildingLevel {
    townHallRequired?: number;
    builderHallRequired?: number;
    districtHallRequired?: number;
    images: {
        normal: string;
    };
}
type OtherBuilding = Building<OtherBuildingLevel>;
interface HomeOtherBuilding extends Building<OtherBuildingLevel> {
    availablePerTownHall: TownHallAvailability[];
}
interface HomeHelperHutBuildingLevel extends BuildingLevel {
    townHallRequired: number;
    images: {
        normal: string;
        active: string;
    };
}
interface HomeHelperHutBuilding extends Building<HomeHelperHutBuildingLevel> {
    availablePerTownHall: TownHallAvailability[];
}
interface HomeHelperLevel {
    level: number;
    townHallRequired: number;
    upgradeCost: number;
}
interface HomeHelper<L extends HomeHelperLevel = HomeHelperLevel> {
    id: string;
    name: string;
    description?: string;
    base: 'home';
    category: 'other';
    recruitmentCost: number;
    recruitmentCostResource: 'Gems' | 'Season Challenge Points';
    townHallRequired: number;
    images: {
        normal: string;
    };
    levels: L[];
}
interface HomeWorkRateHelperLevel extends HomeHelperLevel {
    workRate: number;
}
interface HomeAlchemistHelperLevel extends HomeHelperLevel {
    goldElixirConversionMax: number;
    darkElixirConversionMax: number;
    conversionBonusPercent: number;
}
interface HomeProspectorHelperLevel extends HomeHelperLevel {
    shinyOreConversionMax: number;
    glowyOreConversionMax: number;
    starryOreConversionMax: number;
}
type HomeLabAssistantHelper = HomeHelper<HomeWorkRateHelperLevel>;
type HomeBuilderApprenticeHelper = HomeHelper<HomeWorkRateHelperLevel>;
type HomeAlchemistHelper = HomeHelper<HomeAlchemistHelperLevel>;
type HomeProspectorHelper = HomeHelper<HomeProspectorHelperLevel>;
interface BuilderClockTowerBuildingLevel extends BuildingLevel {
    boostDurationMinutes: number;
    timeGainedMinutes: number;
    builderHallRequired: number;
    images: {
        normal: string;
    };
}
interface BuilderClockTowerBuilding extends Building<BuilderClockTowerBuildingLevel> {
    availablePerBuilderHall: BuilderHallAvailability[];
}
type BuilderOtherBuilding = BuilderClockTowerBuilding;
interface ClanCapitalHouseLevel extends BuildingLevel {
    images: {
        normal: string;
        ruin: string;
    };
}
interface ClanCapitalHouse extends Building<ClanCapitalHouseLevel> {
    availablePerDistrict: DistrictAvailability[];
}

interface ResourceBuildingLevel extends BuildingLevel {
    capacity: number;
    productionRate?: number;
    images: {
        normal: string;
    };
}
interface HomeResourceBuildingLevel extends ResourceBuildingLevel {
    townHallRequired: number;
    supercharge?: boolean;
}
interface HomeResourceBuilding extends Building<HomeResourceBuildingLevel> {
    availablePerTownHall: TownHallAvailability[];
}
interface BuilderResourceBuildingLevel extends ResourceBuildingLevel {
    builderHallRequired: number;
}
interface BuilderResourceBuilding extends Building<BuilderResourceBuildingLevel> {
    availablePerBuilderHall: BuilderHallAvailability[];
}
interface ClanCapitalResourceBuildingLevel extends ResourceBuildingLevel {
    districtHallRequired: number;
}
type ClanCapitalResourceBuilding = Building<ClanCapitalResourceBuildingLevel>;

interface HomePetLevel {
    level: number;
    damagePerSecond?: number;
    damagePerHit?: number;
    damageVsWalls?: number;
    healingPerSecond?: number;
    healingPerPulse?: number;
    frostmitesPerSummon?: number;
    maxFrostmites?: number;
    stunDuration?: number;
    reviveDuration?: number;
    invisibilityDuration?: number;
    brainwashDuration?: number;
    poisonMaxDps?: number;
    poisonSpeedDecreasePercent?: number;
    poisonAttackRateDecreasePercent?: number;
    dpsOnResourceBuildings?: number;
    hitpoints: number;
    petHouseLevelRequired: number;
    townHallRequired: number;
    upgradeCost: number;
    upgradeCostResource: ResourceType;
    upgradeTime: BuildTime;
}
interface HomePet {
    id: string;
    name: string;
    description?: string;
    base: 'home';
    category: 'pet';
    targetType: 'ground' | 'air' | 'both';
    preferredTarget?: string;
    attackType: string;
    movementSpeed: number;
    attackSpeed: number;
    range: number;
    petHouseLevelRequired: number;
    specialAbility?: string;
    numberOfTargets?: number;
    chainDamageDecay?: number;
    rageDuration?: number;
    summonCooldown?: number;
    rageSpeedIncrease?: number;
    rageDamageIncreasePercent?: number;
    maxBoogersSummoned?: number;
    images: {
        icon: string;
        normal: string;
        egg?: string;
    };
    levels: HomePetLevel[];
}

interface SiegeMachineLevel {
    level: number;
    damagePerSecond?: number;
    damagePerHit?: number;
    damageVsWalls?: number;
    damageWhenDestroyed?: number;
    damageWhenDestroyedHitbox1?: number;
    damageWhenDestroyedHitbox2?: number;
    pointBlankDamage?: number;
    flameMaxDps?: number;
    pekkasSpawned?: number;
    wizardsSpawned?: number;
    barrelCount?: number;
    troopLevel?: number;
    troopsSpawnedPerBarrel?: {
        barbarians?: number;
        archers?: number;
        giants?: number;
        wallBreakers?: number;
    };
    lifetime?: number;
    hitpoints: number;
    laboratoryRequired: number;
    townHallRequired: number;
    researchCost: number;
    researchCostResource: ResourceType;
    researchTime: BuildTime;
    images: {
        normal: string;
    };
}
interface SiegeMachine {
    id: string;
    name: string;
    description?: string;
    base: 'home';
    category: 'siege-machine';
    housingSpace: number;
    workshopLevelRequired: number;
    donationCost: DonationCost;
    preferredTarget?: string;
    attackType?: string;
    movementSpeed: number;
    attackSpeed?: number;
    range?: number;
    shotsPerBurst?: number;
    timeBetweenBursts?: number;
    lifetime?: number;
    hpDecayPerSecond?: number;
    images: {
        icon: string;
    };
    levels: SiegeMachineLevel[];
}

interface HomeSpellLevel {
    level: number;
    damage?: number;
    totalHealing?: number;
    healingPerPulse?: number;
    totalHealingOnHeroes?: number;
    damageIncrease?: number;
    speedIncrease?: number;
    spellDuration?: number;
    clonedCapacity?: number;
    recalledCapacity?: number;
    heroHealPercent?: number;
    totemHitpoints?: number;
    maxDamagePerSecond?: number;
    speedDecrease?: number;
    attackRateDecrease?: number;
    radius?: number;
    buildingDamagePercent?: number;
    troopDamagePercent?: number;
    skeletonsGenerated?: number;
    batsGenerated?: number;
    incomingDamageReduction?: number;
    laboratoryRequired: number;
    townHallRequired: number;
    researchCost: number;
    researchCostResource: ResourceType;
    researchTime: BuildTime;
}
interface SkeletonStats {
    preferredTarget: string;
    attackType: string;
    movementSpeed: number;
    attackSpeed: number;
    range: number;
    damagePerSecond: number;
    hitpoints: number;
    armorHitpoints: number;
}
interface BatStats {
    preferredTarget: string;
    attackType: string;
    movementSpeed: number;
    attackSpeed: number;
    range: number;
    damagePerSecond: number;
    damageVsResources: number;
    hitpoints: number;
}
interface HomeSpell {
    id: string;
    name: string;
    description?: string;
    base: 'home';
    category: 'spell';
    spellType: 'regular' | 'dark';
    skeletonStats?: SkeletonStats;
    batStats?: BatStats;
    radius?: number;
    housingSpace: number;
    donationCost: DonationCost;
    spellFactoryLevelRequired: number;
    stunTime?: number;
    numberOfPulses?: number;
    timeBetweenPulses?: number;
    targetType?: 'ground' | 'air' | 'both' | 'defenses';
    boostTime?: number;
    spellDuration?: number;
    clonedLifespan?: number;
    hitpointDecayPerSecond?: number;
    images: {
        icon: string;
    };
    levels: HomeSpellLevel[];
}
interface ClanCapitalSpellLevel {
    level: number;
    districtHallRequired: number;
    radius?: number;
    damage?: number;
    damageIncrease?: number;
    speedIncrease?: number;
    healingPerSecond?: number;
    healingPerPulse?: number;
    skeletonCount?: number;
    images: {
        normal: string;
    };
}
interface ClanCapitalSkeletonStats {
    preferredTarget: string;
    targetsType: 'ground' | 'both';
    damagePerSecond: number;
    damagePerHit: number;
    hitpoints: number;
    shieldHitpoints?: number;
    attackSpeed: number;
    movementSpeed: number;
}
interface ClanCapitalSpell {
    id: string;
    name: string;
    description?: string;
    base: 'clan_capital';
    category: 'spell';
    radius?: number;
    housingSpace: number;
    durationAttacks?: number;
    timeBetweenPulses?: number;
    stunDuration?: number;
    requiredSpellFactory?: string;
    skeletonStatsGround?: ClanCapitalSkeletonStats;
    skeletonStatsAir?: ClanCapitalSkeletonStats;
    levels: ClanCapitalSpellLevel[];
}

interface HomeTroopLevelStats {
    dps?: number;
    damagePerShot?: number;
    dpsOnHeroes?: number;
    deathDamage?: number;
    chainDamagePerShot?: number;
    healingPerSecond?: number;
    healingPerPulse?: number;
    healingPerSecondOnHeroes?: number;
    healingPerPulseOnHeroes?: number;
}
interface HomeTroopLevel {
    level: number;
    hitpoints: number;
    townHallRequired: number;
    laboratoryRequired: number;
    researchCost: number;
    researchCostResource: ResourceType;
    researchTime: BuildTime;
    golemitesSpawned?: number;
    skeletonsPerSummon?: number;
    maxSkeletons?: number;
    skeletonLevel?: number;
    pupsOnOffense?: number;
    pupsOnDefense?: number;
    deathFreezeTimeOnOffense?: number;
    deathFreezeTimeOnDefense?: number;
    firemitesSpawned?: number;
    poisonMaxDps?: number;
    poisonSpeedDecrease?: number;
    poisonAttackRateDecrease?: number;
    auraHpIncrease?: number;
    stats: {
        normal: HomeTroopLevelStats;
        enraged?: HomeTroopLevelStats;
        aura?: HomeTroopLevelStats;
        wall?: HomeTroopLevelStats;
    };
    images: {
        normal: string;
    };
}
interface HomeSuperTroopLevel {
    level: number;
    hitpoints: number;
    townHallRequired?: number;
    pupsOnOffense?: number;
    pupsOnDefense?: number;
    freezeTime?: number;
    stats: {
        normal: HomeTroopLevelStats;
        stage2?: HomeTroopLevelStats;
        stage3?: HomeTroopLevelStats;
    };
    images: {
        normal: string;
    };
}
interface HomeSuperTroop {
    id: string;
    name: string;
    description?: string;
    housingSpace: number;
    movementSpeed: number;
    attackSpeed: number;
    shotsPerBurst?: number;
    timeBetweenBursts?: number;
    range: number;
    boostCost: number;
    boostCostResource: 'Dark Elixir';
    boostDuration: BuildTime;
    regularLevelRequired: number;
    specialAbility?: string;
    images: {
        icon: string;
    };
    levels: HomeSuperTroopLevel[];
}
interface HomeTroop {
    id: string;
    name: string;
    description?: string;
    base: 'home';
    category: 'troop';
    troopType: 'regular' | 'dark';
    housingSpace: number;
    movementSpeed: number;
    range?: number;
    attackSpeed?: number;
    damageType: 'single' | 'splash' | 'area' | 'chain' | 'none';
    targetType: 'ground' | 'air' | 'both';
    barrackLevelRequired: number;
    donationCost: DonationCost;
    lifetime?: number;
    auraAttackSpeed?: number;
    auraRange?: number;
    wallAttackSpeed?: number;
    summonCooldown?: number;
    numberOfTargets?: number;
    evolveTime?: number;
    preferredTarget?: string;
    freezeRadiusOnOffense?: number;
    freezeRadiusOnDefense?: number;
    specialAbility?: string;
    images: {
        icon: string;
    };
    levels: HomeTroopLevel[];
    superTroop?: HomeSuperTroop;
}
interface TroopModeStats {
    dps: number;
    damagePerShot: number;
    trainingCost: number;
    trainingCostResource: ResourceType;
    trainingTime: number;
    movementSpeed: number;
}
interface BuilderTroopLevel {
    level: number;
    hitpoints: number;
    dps?: number;
    damagePerShot?: number;
    unitsPerCamp: number;
    deathDamage?: number;
    skeletonBombDamage?: number;
    skeletonBombSkeletons?: number;
    cloakDurationSeconds?: number;
    rageDurationSeconds?: number;
    boxerBlockDurationSeconds?: number;
    powerPunchDamage?: number;
    powerShotAttacks?: number;
    powerShotDamagePerHit?: number;
    damageVsWalls?: number;
    bouncingBombDamage?: number;
    bouncingBombDamageVsWalls?: number;
    tantrumDamageBonus?: number;
    fierySneezeDamageMin?: number;
    fierySneezeDamageMax?: number;
    mortarDps?: number;
    mortarDamagePerShot?: number;
    batSummonCooldown?: number;
    batsPerSummon?: number;
    batsMax?: number;
    batSwarmCount?: number;
    overchargeDamage?: number;
    stunDuration?: number;
    infernoInitialDps?: number;
    infernoDpsAfter1_5s?: number;
    infernoDpsAfter3_0s?: number;
    electroDps?: number;
    electroDamagePerShot?: number;
    researchCost: number;
    researchCostResource: 'Builder Elixir';
    researchTime: BuildTime;
    starLabRequired: number;
    images: {
        normal: string;
        mortarMode?: string;
        electroMode?: string;
    };
}
interface BuilderTroop {
    id: string;
    name: string;
    description?: string;
    base: 'builder';
    category: 'troop';
    housingSpace: number;
    movementSpeed: number;
    range: number;
    attackSpeed?: number;
    damageType: 'single' | 'splash' | 'area';
    targetType: 'ground' | 'air' | 'both';
    builderBarracksRequired?: number;
    specialAbility?: string;
    passiveAbility?: string;
    preferredTarget?: string;
    skeletonsSummoned?: number;
    skeletonsMax?: number;
    skeletonSummonCooldown?: number;
    abilityCooldown?: number;
    electroAttackSpeed?: number;
    electroNumberOfTargets?: number;
    electroChainDamageDecay?: number;
    rageSpeedIncrease?: number;
    rageDamageIncrease?: number;
    wallDamageMultiplier?: number;
    explodingRange?: number;
    mortarMovementSpeed?: number;
    mortarRange?: number;
    mortarAttackSpeed?: number;
    mortarDamageType?: 'single' | 'splash' | 'area';
    levels: BuilderTroopLevel[];
}
interface ClanCapitalTroopStats {
    dps?: number;
    damagePerShot?: number;
    chainDamagePerShot?: number;
    damageVsWalls?: number;
    infernoInitialDps?: number;
    infernoDpsAfter1_7s?: number;
    infernoDpsAfter3_2s?: number;
    drillInitialDps?: number;
    drillDpsAfter1_5s?: number;
    drillDpsAfter3s?: number;
}
interface ClanCapitalTroopLevel {
    level: number;
    hitpoints: number;
    districtHallRequired: number;
    deathDamage?: number;
    spawnedSkeletons?: number;
    spawnedSkeletonGliders?: number;
    spawnedSkeletonsOnDeath?: number;
    lastStandHitpoints?: number;
    stats: {
        normal: ClanCapitalTroopStats;
    };
    images: {
        normal: string;
    };
}
interface ClanCapitalTroopSubUnitLevel {
    level: number;
    hitpoints: number;
    stats: {
        normal: ClanCapitalTroopStats;
    };
    images: {
        normal: string;
    };
}
interface ClanCapitalTroopSubUnit {
    name: string;
    damageType: 'single' | 'splash' | 'area';
    targetType: 'ground' | 'air' | 'both';
    movementSpeed: number;
    attackSpeed?: number;
    range?: number;
    preferredTarget?: string;
    levels: ClanCapitalTroopSubUnitLevel[];
}
interface ClanCapitalTroop {
    id: string;
    name: string;
    description?: string;
    /** Individual unit name when the troop deploys multiple units (e.g. "Minion" for Minion Horde). Absent for single-unit troops. */
    unitName?: string;
    /** Additional units deployed alongside the primary unit, each with their own level data. */
    subUnits?: ClanCapitalTroopSubUnit[];
    base: 'clan_capital';
    category: 'troop';
    damageType: 'single' | 'splash' | 'area';
    targetType: 'ground' | 'air' | 'both';
    housingSpace: number;
    movementSpeed: number;
    attackSpeed?: number;
    range?: number;
    preferredTarget?: string;
    specialAbility?: string;
    cloakDuration?: number;
    rageMovementSpeedIncrease?: number;
    rageDamageIncrease?: number;
    spawnedBarbarians?: number;
    wallDamageMultiplier?: number;
    boostDuration?: number;
    timeBetweenBursts?: number;
    images: {
        icon: string;
    };
    levels: ClanCapitalTroopLevel[];
}

interface TrapLevel {
    level: number;
    damage: number;
    springCapacity?: number;
    damageRadius?: number;
    spawnedUnits?: number;
    skeletonLevel?: number;
    duration?: number;
    buildCost: number;
    buildCostResource: ResourceType;
    buildTime: BuildTime;
    xpGained: number;
    townHallRequired: number;
    images: {
        normal: string;
        air?: string;
    };
}
interface HomeTrap {
    id: string;
    name: string;
    description?: string;
    base: 'home';
    category: 'trap';
    size: string;
    triggerRadius: number;
    triggerHousingSpace?: number;
    damageRadius?: number;
    damageType: 'splash' | 'single';
    targetType: 'ground' | 'air' | 'both';
    favoriteTarget?: string;
    specialAbility?: string;
    levels: TrapLevel[];
    availablePerTownHall: TownHallAvailability[];
}
interface BuilderTrapLevel {
    level: number;
    damage?: number;
    damageVsHeroes?: number;
    springCapacity?: number;
    buildCost: number;
    buildCostResource: ResourceType;
    buildTime: BuildTime;
    xpGained: number;
    builderHallRequired: number;
    images: {
        normal: string;
        air?: string;
    };
}
interface BuilderTrap {
    id: string;
    name: string;
    description?: string;
    base: 'builder';
    category: 'trap';
    size: string;
    triggerRadius: number;
    damageRadius?: number;
    springCapacity?: number;
    aoeRadius?: number;
    pushDistance?: number;
    targetType: 'ground' | 'air' | 'both';
    levels: BuilderTrapLevel[];
    availablePerBuilderHall: BuilderHallAvailability[];
}
interface ClanCapitalTrapLevel {
    level: number;
    damage: number;
    projectileCount?: number;
    buildCost: number;
    buildCostResource: ResourceType;
    buildTime: BuildTime;
    xpGained: number;
    capitalHallRequired: number;
    districtHallRequired: number;
    images: {
        normal: string;
        air?: string;
    };
}
interface ClanCapitalTrap {
    id: string;
    name: string;
    description?: string;
    base: 'clan_capital';
    category: 'trap';
    size: string;
    triggerRadius: number;
    damageRadius?: number;
    damageType: 'splash' | 'single';
    targetType: 'ground' | 'air' | 'both';
    favoriteTarget?: string;
    levels: ClanCapitalTrapLevel[];
    availablePerCapitalHall?: CapitalHallAvailability[];
    availablePerDistrict: DistrictAvailability[];
}

interface WallLevel {
    level: number;
    hitpoints: number;
    buildCost: number;
    buildCostResource: ResourceType;
    wallRings: number;
    buildTime: BuildTime;
    xpGained: number;
    townHallRequired: number;
    images: {
        normal: string;
    };
}
interface HomeWall {
    id: string;
    name: string;
    description?: string;
    base: 'home';
    category: 'wall';
    size: '1x1';
    levels: WallLevel[];
    availablePerTownHall: TownHallAvailability[];
}
interface BuilderWallLevel {
    level: number;
    hitpoints: number;
    buildCost: number;
    buildCostResource: ResourceType;
    wallRings: number;
    buildTime: BuildTime;
    xpGained: number;
    builderHallRequired: number;
    images: {
        normal: string;
    };
}
interface BuilderWall {
    id: string;
    name: string;
    description?: string;
    base: 'builder';
    category: 'wall';
    size: '1x1';
    levels: BuilderWallLevel[];
    availablePerBuilderHall: BuilderHallAvailability[];
}
interface ClanCapitalWallLevel {
    level: number;
    hitpoints: number;
    buildCost: number;
    buildCostResource: ResourceType;
    buildTime: BuildTime;
    xpGained: number;
    capitalHallRequired: number;
    districtHallRequired: number;
    images: {
        normal: string;
        corner: string;
    };
}
interface ClanCapitalWall {
    id: string;
    name: string;
    description?: string;
    base: 'clan_capital';
    category: 'wall';
    size: '1x1';
    levels: ClanCapitalWallLevel[];
    availablePerCapitalHall: CapitalHallAvailability[];
    availablePerDistrict: DistrictAvailability[];
}

type MagicItemType = 'snack' | 'potion' | 'book' | 'hammer' | 'utility';
/** Effect that reduces build or research time by a speed multiplier for 1 hour. */
interface TimeReductionEffect {
    type: 'time-reduction';
    /** Speed multiplier (e.g. 2 = 2x speed → reduces 2 hours in 60 min). */
    multiplier: number;
    /** Duration in hours that the effect is active. */
    durationHours: number;
    /** Which queue the effect applies to. */
    appliesTo: 'builders' | 'research' | 'pets';
}
/** Effect that boosts units/equipment to max Town Hall level for a number of battles. */
interface CombatBoostEffect {
    type: 'combat-boost';
    boostTo: 'max-town-hall-level';
    /** Number of battles the boost is active for. */
    battles: number;
    /** Which unit categories are boosted. */
    appliesTo: string[];
}
/** Effect that enables free Clan Castle reinforcements for a duration. */
interface ClanCastleEffect {
    type: 'clan-castle';
    /** Duration in hours that the effect is active. */
    durationHours: number;
    appliesTo: 'clan-castle-reinforcements';
}
/** Effect that instantly completes any ongoing upgrade of the given type. Used by Books. */
interface InstantCompleteEffect {
    type: 'instant-complete';
    /** Which upgrade queue the book can complete. */
    appliesTo: 'troops' | 'buildings' | 'spells' | 'heroes-and-pets' | 'any';
}
/** Effect that instantly upgrades a unit or building to the next level. Used by Hammers. */
interface InstantUpgradeEffect {
    type: 'instant-upgrade';
    /** Which category of item the hammer applies to. */
    appliesTo: 'troops' | 'buildings' | 'spells' | 'heroes-and-pets';
}
/** Effect that boosts units/heroes to max Town Hall level for a duration. Used by Hero/Power Potions. */
interface UnitLevelBoostEffect {
    type: 'unit-level-boost';
    boostTo: 'max-town-hall-level';
    /** Duration in hours that the boost is active. */
    durationHours: number;
    /** Which unit categories are boosted. */
    appliesTo: string[];
}
/** Effect that boosts resource collectors for a duration. Used by Resource Potion. */
interface ResourceCollectorBoostEffect {
    type: 'resource-collector-boost';
    /** Speed multiplier applied to collectors (e.g. 2 = 2x production rate). */
    multiplier: number;
    /** Duration in days that the boost is active. */
    durationDays: number;
}
/** Effect that activates the Clock Tower boost for a duration. Used by Clock Tower Potion. */
interface ClockTowerBoostEffect {
    type: 'clock-tower-boost';
    /** Duration in minutes that the Clock Tower is boosted. */
    durationMinutes: number;
}
/** Effect that boosts a single troop to its Super Troop version for a duration. Used by Super Potion. */
interface SuperTroopEffect {
    type: 'super-troop';
    /** Duration in days the Super Troop boost is active. */
    durationDays: number;
}
/** Effect that upgrades a Wall piece without consuming resources. Used by Wall Ring. */
interface WallUpgradeEffect {
    type: 'wall-upgrade';
    /** Gold/Elixir equivalent per ring in Home Village. */
    homeVillageGoldEquivalent: number;
    /** Builder Gold/Elixir equivalent per ring in Builder Base. */
    builderBaseGoldEquivalent: number;
}
/** Effect that makes a single obstacle permanently movable. Used by Shovel of Obstacles. */
interface ObstacleMoveEffect {
    type: 'obstacle-move';
    /** What the shovel can be applied to. */
    targets: 'single-obstacle';
}
/** Effect that instantly resets the Star Bonus availability in the Builder Base. Used by Builder Star Jar. */
interface StarBonusResetEffect {
    type: 'star-bonus-reset';
    appliesTo: 'builder-base-star-bonus';
}
type MagicItemEffect = TimeReductionEffect | CombatBoostEffect | ClanCastleEffect | InstantCompleteEffect | InstantUpgradeEffect | UnitLevelBoostEffect | ResourceCollectorBoostEffect | ClockTowerBoostEffect | SuperTroopEffect | WallUpgradeEffect | ObstacleMoveEffect | StarBonusResetEffect;
interface MagicItem {
    id: string;
    name: string;
    description: string;
    itemType: MagicItemType;
    effect: MagicItemEffect;
    maxCapacity?: number;
    sellingPrice?: number;
    image: string;
}
type MagicSnack = MagicItem & {
    itemType: 'snack';
};
type MagicPotion = MagicItem & {
    itemType: 'potion';
};
type MagicBook = MagicItem & {
    itemType: 'book';
};
type MagicHammer = MagicItem & {
    itemType: 'hammer';
};
type MagicUtility = MagicItem & {
    itemType: 'utility';
};

/** Valid Builder Boost / Research Boost tier percentages. */
type BoostTier = 10 | 15 | 20;
/** Resource types that can have their build or research cost reduced by a boost. */
type BuildCostResource = 'Gold' | 'Elixir' | 'Dark Elixir';
/** Valid Clock Tower levels (1–10). */
type ClockTowerLevel = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10;
/** Valid Lab Assistant levels (1–12). */
type LabAssistantLevel = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;
/** Valid Builder's Apprentice levels (1–8). */
type BuildersApprenticeLevel = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8;
/** Valid Alchemist levels (1–7). */
type AlchemistLevel = 1 | 2 | 3 | 4 | 5 | 6 | 7;
/** Valid Prospector levels (currently only level 1). */
type ProspectorLevel = 1;
/** Result returned by `calculators().helpers().alchemist()`. */
interface AlchemistResult {
    /** Amount of Gold/Elixir actually used (capped at the level's conversion max). */
    input: number;
    /** Dark Elixir before the bonus is applied. */
    base: number;
    /** Bonus Dark Elixir from the Alchemist's conversionBonusPercent. */
    bonus: number;
    /** Total Dark Elixir received (base + bonus). */
    total: number;
}
/** Result returned by `calculators().helpers().prospector()`. */
interface ProspectorResult {
    shinyOre: number;
    glowyOre: number;
    starryOre: number;
}

/** A single challenge task type that can appear in Season Pass challenges. */
interface SeasonPassChallenge {
    id: string;
    /** Human-readable task description, e.g. "Start any building upgrade". */
    name: string;
    /** Clarifying notes and edge cases for this challenge type. */
    notes: string[];
    /** Path to the challenge icon image. */
    image: string;
}

/** Badge tier displayed on a clan's profile. */
type ClanBadge = 'bronze' | 'silver' | 'gold' | 'crystal' | 'master' | 'champion' | 'titan' | 'legend';
/** Donation limits unlocked at a given clan level. */
interface ClanDonationLimit {
    troops: number;
    spells: number;
    siegeMachines: number;
}
/** Perks that apply at a given clan level. */
interface ClanLevelPerks {
    /** Maximum troops/spells/siege machines that can be donated per request. */
    donationLimit: ClanDonationLimit;
    /** Number of extra levels donated troops/spells are upgraded by (0, 1, or 2). */
    donationUpgradeLevels: number;
    /** Extra loot stored in the Treasury, as a percentage (0–50). */
    treasuryExtraStorage: number;
    /** Extra loot earned from the war bonus, as a percentage (0–25). */
    warBonusExtraLoot: number;
}
/** A single clan level entry with XP requirements and unlocked perks. */
interface ClanLevel {
    level: number;
    /** XP required to reach this level from the previous one. Null for level 1. */
    xpRequired: number | null;
    /** Total XP accumulated to reach this level. Null for level 1. */
    cumulativeXp: number | null;
    badge: ClanBadge;
    /** Path to the badge image. */
    image: string;
    perks: ClanLevelPerks;
}
/** A clan label that can be displayed on a clan's profile. */
interface ClanLabel {
    id: string;
    name: string;
    /** Path to the label icon image. */
    image: string;
}
/** Gold/Elixir and Dark Elixir amounts. */
interface WarLootAmount {
    goldAndElixir: number;
    darkElixir: number;
}
/** Maximum available war base loot for a given Town Hall level. */
interface WarBaseLootEntry {
    townHallLevel: number;
    goldAndElixir: number;
    darkElixir: number;
}
/** War bonus (victory/draw/defeat) for a specific Town Hall level. */
interface WarBonusByTownHall {
    townHallLevel: number;
    victory: WarLootAmount;
    draw: WarLootAmount;
    defeat: WarLootAmount;
}
/** War bonus amounts for a clan level tier across all Town Hall levels. */
interface WarBonusTier {
    /** E.g. "1-2", "3-4", "9+". */
    clanLevelRange: string;
    label: string;
    minClanLevel: number;
    /** Absent for the top tier (9+). */
    maxClanLevel?: number;
    byTownHall: WarBonusByTownHall[];
}
/** Maximum ore available in an enemy war base for a given Town Hall level. */
interface WarBaseOreEntry {
    /** Number for TH8–15; string "16-18" for the grouped top tier. */
    townHallLevel: number | string;
    shinyOre: number;
    glowyOre: number;
    /** Null for TH8–9 (Starry Ore not available). */
    starryOre: number | null;
}

/** The village or base an achievement belongs to. */
type AchievementBase = 'home' | 'builder' | 'clan-capital';
/** A single tier within an achievement. */
interface AchievementTier {
    tier: number;
    /** The threshold value required to complete this tier. */
    requirement: number;
    xpRewarded: number;
    gemsRewarded: number;
}
/** A single in-game achievement with one or more tiers. */
interface Achievement {
    id: string;
    name: string;
    /** Which village/base this achievement belongs to. */
    base: AchievementBase;
    /** Description of what progress is tracked (e.g. "Total Gold looted"). */
    dataInvolved: string;
    tiers: AchievementTier[];
}

/** A ranked battles league group name (e.g. "Skeleton", "Electro"). */
type LeagueGroup = 'Unranked' | 'Skeleton' | 'Barbarian' | 'Archer' | 'Wizard' | 'Valkyrie' | 'Witch' | 'Golem' | 'PEKKA' | 'Titan' | 'Dragon' | 'Electro' | 'Legend';
/** A single ranked battles league (e.g. "Skeleton 1", "Legend"). */
interface RankedBattleLeague {
    /** Kebab-case identifier, e.g. `"skeleton-1"`, `"pekka-22"`, `"legend"`. */
    id: string;
    /** Display name, e.g. `"Skeleton 1"`, `"P.E.K.K.A 22"`, `"Legend"`. */
    name: string;
    /** The group this league belongs to. */
    leagueGroup: LeagueGroup;
    /** Sequential number within the group (1–33). Null for Legend and Unranked. */
    leagueNumber: number | null;
    /** Path to the league badge image. */
    image: string;
    /** Number of attacks and defenses per weekly tournament. Null for Legend and Unranked. */
    attacksPerWeek: number | null;
    /** Percentage of players promoted at the end of the week. Null for the top league and Unranked. */
    percentPromoted: number | null;
    /** Percentage of players demoted at the end of the week. Null for the bottom league and Unranked. */
    percentDemoted: number | null;
}
/** The minimum league a player cannot be demoted below, based on Town Hall level. */
interface LeagueFloorEntry {
    townHallLevel: number;
    /** ID of the floor league (e.g. `"skeleton-1"`). */
    leagueId: string;
}
/** A difficulty modifier applied to high-ranked leagues. */
interface DifficultyModifier {
    /** ID of the league this modifier applies to. */
    leagueId: string;
    /** Modifier tier name: "Expert", "Master", or "Legend". */
    modifier: 'Expert' | 'Master' | 'Legend';
    /** Percentage bonus applied to all defense DPS. */
    defenseDpsBonus: number;
    /** Percentage bonus applied to defending hero DPS and HP. */
    defendingHeroDpsHpBonus: number;
    /** Percentage penalty applied to attacking hero DPS and HP. Null when not applicable. */
    attackingHeroDpsHpPenalty: number | null;
}
/** Combined Gold/Elixir and Dark Elixir loot amounts. */
interface RankedBattleLootAmount {
    /** Gold and Elixir (same value). Null for Unranked (no available loot). */
    goldAndElixir: number | null;
    /** Dark Elixir amount. Null for Unranked. */
    darkElixir: number | null;
}
/** Ore rewards from the weekly Star Bonus. */
interface StarBonusOre {
    goldAndElixir: number | null;
    darkElixir: number | null;
    shinyOre: number | null;
    glowyOre: number | null;
    starryOre: number | null;
}
/**
 * League bonus and star bonus for Town Halls below 7.
 * These Town Halls cannot participate in ranked leagues but still receive bonuses.
 */
interface LowerThBonus {
    /** The Town Hall level (2–6). */
    townHallLevel: number;
    /** Maximum league bonus in Gold/Elixir (no dark elixir at these Town Hall levels). */
    maxLeagueBonus: number;
    /** Weekly Star Bonus in Gold/Elixir. */
    starBonus: number;
}
/** Loot available for a specific league at a given Town Hall level. */
interface RankedBattleLootEntry {
    /** ID of the league this entry applies to. */
    leagueId: string;
    /**
     * Whether this league is below the Town Hall's league floor.
     * Below-floor leagues are shown in the game UI but cannot be reached in practice.
     */
    underfloor: boolean;
    /** Maximum loot available to steal from the opponent's base. */
    maxAvailableLoot: RankedBattleLootAmount;
    /** Maximum bonus earned from winning a ranked battle in this league. */
    maxLeagueBonus: RankedBattleLootAmount;
    /** Weekly Star Bonus rewards for completing 8 stars in this league. */
    starBonus: StarBonusOre;
}

/** A Clan Capital league group name. */
type ClanCapitalLeagueGroup = 'Unranked' | 'Bronze' | 'Silver' | 'Gold' | 'Crystal' | 'Master' | 'Champion' | 'Titan' | 'Legend';
/** A single Clan Capital Raid Weekend league entry. */
interface ClanCapitalLeague {
    /** Kebab-case identifier, e.g. `"bronze-iii"`, `"legend"`. */
    id: string;
    /** Display name, e.g. `"Bronze III"`, `"Legend"`. */
    name: string;
    /** The group this league belongs to. */
    group: ClanCapitalLeagueGroup;
    /**
     * Tier within the group (1 = I, 2 = II, 3 = III).
     * Null for Unranked and Legend which have no tier.
     */
    tier: 1 | 2 | 3 | null;
    /** Minimum Capital Trophies required to be in this league. */
    trophyMin: number;
    /** Maximum Capital Trophies for this league. Null for Legend (5000+). */
    trophyMax: number | null;
    /** Clan XP awarded to the clan at the end of a Raid Weekend in this league. */
    clanXpEarned: number;
    /** Path to the league badge image. */
    image: string;
}

/** A Builder Base league group name. */
type BuilderBaseLeagueGroup = 'Wood' | 'Clay' | 'Stone' | 'Copper' | 'Brass' | 'Iron' | 'Steel' | 'Titanium' | 'Platinum' | 'Emerald' | 'Ruby' | 'Diamond';
/** Star Bonus for a Builder Base league. Earned once per day by reaching the required star count. */
interface BuilderBaseStarBonus {
    /** Number of stars that must be earned to claim the bonus. */
    starsRequired: number;
    /** Amount of Builder Gold and Builder Elixir each awarded when claimed. */
    reward: number;
}
/**
 * Battle result rewards for a given star outcome in a Builder Base league.
 * Gold is earned by the attacker; Elixir is earned by the defender.
 */
interface BuilderBaseBattleResult {
    /** Builder Gold earned by the attacker. */
    attackerGold: number;
    /** Builder Elixir earned by the defender. */
    defenderElixir: number;
}
/** A single Builder Base league entry. */
interface BuilderBaseLeague {
    /** Kebab-case identifier, e.g. `"wood-v"`, `"diamond"`. */
    id: string;
    /** Display name, e.g. `"Wood V"`, `"Diamond"`. */
    name: string;
    /** The group this league belongs to. */
    group: BuilderBaseLeagueGroup;
    /**
     * Tier within the group (1 = I, 2 = II, 3 = III, 4 = IV, 5 = V).
     * Null for Diamond which has no tier.
     */
    tier: 1 | 2 | 3 | 4 | 5 | null;
    /** Minimum Builder Base trophies required to be in this league. */
    trophyMin: number;
    /** Maximum Builder Base trophies for this league. Null for Diamond (6200+). */
    trophyMax: number | null;
    /** Path to the league badge image. */
    image: string;
    /** Star Bonus earned once per day by reaching the required star count. */
    starBonus: BuilderBaseStarBonus;
    /**
     * Battle result rewards indexed by star count (0–6).
     * `battleResults[0]` = 0-star outcome, `battleResults[6]` = 6-star outcome.
     */
    battleResults: [
        BuilderBaseBattleResult,
        BuilderBaseBattleResult,
        BuilderBaseBattleResult,
        BuilderBaseBattleResult,
        BuilderBaseBattleResult,
        BuilderBaseBattleResult,
        BuilderBaseBattleResult
    ];
}

interface ClanCapitalDailyForgeEntry {
    /** Town Hall level (6–18). */
    townHallLevel: number;
    /** Capital Gold obtained from the daily forge at this Town Hall level. */
    capitalGoldObtained: number;
}
interface ClanCapitalAvailableForgeEntry {
    /** Number of manual forge slots available. */
    slots: number;
    /** Town Hall level required to unlock this many slots. */
    townHallRequired: number;
}
/** Resource conversion entry for home-village Auto Forge or Forge (keyed by Town Hall level). */
interface ClanCapitalHomeForgeEntry {
    /** Town Hall level (7–18). */
    townHallLevel: number;
    /** Gold or Elixir cost (standard). */
    goldElixirCost: number;
    /** Gold or Elixir cost with Gold Pass discount (Auto Forge only). */
    goldElixirCostGoldPass?: number;
    /** Dark Elixir cost (standard). Present from TH13+. */
    darkElixirCost?: number;
    /** Dark Elixir cost with Gold Pass discount (Auto Forge only). Present from TH13+. */
    darkElixirCostGoldPass?: number;
    /** Capital Gold obtained from this conversion. */
    capitalGoldObtained: number;
}
/** Resource conversion entry for builder-base Auto Forge or Forge (keyed by Builder Hall level). */
interface ClanCapitalBuilderForgeEntry {
    /** Builder Hall level (8–10). */
    builderHallLevel: number;
    /** Builder Gold or Builder Elixir cost (standard). */
    builderGoldElixirCost: number;
    /** Builder Gold or Builder Elixir cost with Gold Pass discount (Auto Forge only). */
    builderGoldElixirCostGoldPass?: number;
    /** Capital Gold obtained from this conversion. */
    capitalGoldObtained: number;
}
interface ClanCapitalForgeRates {
    home: ClanCapitalHomeForgeEntry[];
    builder: ClanCapitalBuilderForgeEntry[];
}
interface ClanCapitalForgeData {
    dailyForge: ClanCapitalDailyForgeEntry[];
    craftingTime: BuildTime;
    availableForges: ClanCapitalAvailableForgeEntry[];
    autoForge: ClanCapitalForgeRates;
    forge: ClanCapitalForgeRates;
}

/**
 * Abstract base class providing the five standard terminal methods shared by all query builders.
 * Subclasses hold a typed data array and implement domain-specific filter and sort methods.
 */
declare abstract class QueryBase<T extends {
    id: string;
    name: string;
}> {
    protected readonly data: T[];
    constructor(data: T[]);
    /** Return all results as an array. */
    get(): T[];
    /** Return the first result, or `undefined` if there are none. */
    first(): T | undefined;
    /** Find an item by its exact ID. */
    find(id: string): T | undefined;
    /** Find an item by name (case-insensitive exact match). */
    findByName(name: string): T | undefined;
    /** Return the number of results. */
    count(): number;
}

/** Query class for the Barracks building. Returned by `home().armyBuildings().barracks()`. */
declare class HomeVillageBarracks extends QueryBase<HomeBarracksBuilding> {
    constructor(data?: HomeBarracksBuilding[]);
}
/** Query class for the Dark Barracks building. Returned by `home().armyBuildings().darkBarracks()`. */
declare class HomeVillageDarkBarracks extends QueryBase<HomeBarracksBuilding> {
    constructor(data?: HomeBarracksBuilding[]);
}
/** Query class for the Laboratory building. Returned by `home().armyBuildings().laboratory()`. */
declare class HomeVillageLaboratory extends QueryBase<HomeResearchBuilding> {
    constructor(data?: HomeResearchBuilding[]);
}
/** Query class for the Spell Factory building. Returned by `home().armyBuildings().spellFactory()`. */
declare class HomeVillageSpellFactory extends QueryBase<HomeSpellFactoryBuilding> {
    constructor(data?: HomeSpellFactoryBuilding[]);
}
/** Query class for the Dark Spell Factory building. Returned by `home().armyBuildings().darkSpellFactory()`. */
declare class HomeVillageDarkSpellFactory extends QueryBase<HomeSpellFactoryBuilding> {
    constructor(data?: HomeSpellFactoryBuilding[]);
}
/** Query class for the Hero Banner building. Returned by `home().armyBuildings().heroBanner()`. */
declare class HomeVillageHeroBanner extends QueryBase<HeroBannerBuilding> {
    constructor(data?: HeroBannerBuilding[]);
}
/** Query class for the Hero Hall building. Returned by `home().armyBuildings().heroHall()`. */
declare class HomeVillageHeroHall extends QueryBase<HomeHeroHallBuilding> {
    constructor(data?: HomeHeroHallBuilding[]);
}
/** Query class for the Blacksmith building. Returned by `home().armyBuildings().blacksmith()`. */
declare class HomeVillageBlacksmith extends QueryBase<HomeBlacksmithBuilding> {
    constructor(data?: HomeBlacksmithBuilding[]);
}
/** Query class for the Workshop building. Returned by `home().armyBuildings().workshop()`. */
declare class HomeVillageWorkshop extends QueryBase<HomeWorkshopBuilding> {
    constructor(data?: HomeWorkshopBuilding[]);
}
/** Query class for the Pet House building. Returned by `home().armyBuildings().petHouse()`. */
declare class HomeVillagePetHouse extends QueryBase<HomePetHouseBuilding> {
    constructor(data?: HomePetHouseBuilding[]);
}
/**
 * Query class for all Home Village army buildings.
 * Returned by `home().armyBuildings()`.
 *
 * Note: the generic `get()` / `count()` methods only include Army Camp (typed as `HomeArmyBuilding`).
 * Buildings with specific types (Barracks, Laboratory, etc.) are accessed via their own accessor methods.
 */
declare class HomeVillageArmyBuildings extends QueryBase<HomeArmyBuilding> {
    constructor(data?: HomeArmyBuilding[]);
    armyCamp(): HomeVillageArmyBuildings;
    barracks(): HomeVillageBarracks;
    darkBarracks(): HomeVillageDarkBarracks;
    laboratory(): HomeVillageLaboratory;
    spellFactory(): HomeVillageSpellFactory;
    darkSpellFactory(): HomeVillageDarkSpellFactory;
    heroBanner(): HomeVillageHeroBanner;
    heroHall(): HomeVillageHeroHall;
    blacksmith(): HomeVillageBlacksmith;
    workshop(): HomeVillageWorkshop;
    petHouse(): HomeVillagePetHouse;
    /** Filter to army buildings available (count > 0) at the given Town Hall level. */
    byTownHall(level: number): HomeVillageArmyBuildings;
}

/**
 * Query class for all Home Village crafted defenses.
 * Returned by `home().craftedDefenses()`.
 */
declare class HomeVillageCraftedDefenses extends QueryBase<CraftedDefense> {
    constructor(data?: CraftedDefense[]);
    hookTower(): HomeVillageCraftedDefenses;
    flameSpinner(): HomeVillageCraftedDefenses;
    crusherMortar(): HomeVillageCraftedDefenses;
    heroBell(): HomeVillageCraftedDefenses;
    bombHive(): HomeVillageCraftedDefenses;
    lightBeam(): HomeVillageCraftedDefenses;
    roaster(): HomeVillageCraftedDefenses;
    airBombs(): HomeVillageCraftedDefenses;
    lavaLauncher(): HomeVillageCraftedDefenses;
    /** Filter to a specific crafting phase (1, 2, 3, …). */
    byPhase(phase: number): HomeVillageCraftedDefenses;
    /** Filter to defenses in the currently active crafting phase. */
    current(): HomeVillageCraftedDefenses;
    /** Filter to defenses from previous (retired) crafting phases. */
    former(): HomeVillageCraftedDefenses;
    /** Filter by target type. */
    byTargetType(type: CraftedDefense['targetType']): HomeVillageCraftedDefenses;
}

/**
 * Query class for all Home Village stationary defenses.
 * Returned by `home().defenses()`.
 */
declare class HomeVillageDefenses extends QueryBase<HomeDefense> {
    constructor(data?: HomeDefense[]);
    cannon(): HomeVillageDefenses;
    archerTower(): HomeVillageDefenses;
    mortar(): HomeVillageDefenses;
    airDefense(): HomeVillageDefenses;
    wizardTower(): HomeVillageDefenses;
    airSweeper(): HomeVillageDefenses;
    hiddenTesla(): HomeVillageDefenses;
    bombTower(): HomeVillageDefenses;
    xBow(): HomeVillageDefenses;
    infernoTower(): HomeVillageDefenses;
    eagleArtillery(): HomeVillageDefenses;
    scattershot(): HomeVillageDefenses;
    buildersHut(): HomeVillageDefenses;
    spellTower(): HomeVillageDefenses;
    monolith(): HomeVillageDefenses;
    multiArcherTower(): HomeVillageDefenses;
    firespitter(): HomeVillageDefenses;
    multiGearTower(): HomeVillageDefenses;
    revengeTower(): HomeVillageDefenses;
    ricochetCannon(): HomeVillageDefenses;
    superWizardTower(): HomeVillageDefenses;
    craftingStation(): HomeVillageDefenses;
    /** Filter to a specific building by name (case-insensitive). */
    byBuilding(name: string): HomeVillageDefenses;
    /**
     * Filter to buildings that have at least one level available at or before
     * the given Town Hall level.
     */
    byTownHall(level: number): HomeVillageDefenses;
    /** Filter by damage type of the normal mode (use 'none' for knockback-only buildings like Air Sweeper). Buildings without a normal mode (e.g. Spell Tower) are excluded. */
    byDamageType(type: DefenseMode['damageType']): HomeVillageDefenses;
    /** Filter to buildings that can be geared up (any gear-up mode). */
    hasGearUp(): HomeVillageDefenses;
}

/**
 * Query class for all Home Village Guardians.
 * Returned by `home().guardians()`.
 */
declare class HomeVillageGuardians extends QueryBase<Guardian> {
    constructor(data?: Guardian[]);
    longshot(): HomeVillageGuardians;
    smasher(): HomeVillageGuardians;
    /** Filter to a specific guardian variant by type (e.g. 'longshot', 'smasher'). */
    byGuardianType(type: string): HomeVillageGuardians;
    /**
     * Filter to guardians that have at least one level available at or before
     * the given Town Hall level.
     */
    byTownHall(level: number): HomeVillageGuardians;
}

/**
 * Query class for all Home Village hero equipment.
 * Returned by `home().heroEquipment()`.
 */
declare class HomeVillageHeroEquipment extends QueryBase<HeroEquipment> {
    constructor(data?: HeroEquipment[]);
    archerPuppet(): HomeVillageHeroEquipment;
    giantArrow(): HomeVillageHeroEquipment;
    barbarianPuppet(): HomeVillageHeroEquipment;
    rageVial(): HomeVillageHeroEquipment;
    electroBoots(): HomeVillageHeroEquipment;
    earthquakeBoots(): HomeVillageHeroEquipment;
    vampstache(): HomeVillageHeroEquipment;
    giantGauntlet(): HomeVillageHeroEquipment;
    stunBlaster(): HomeVillageHeroEquipment;
    spikyBall(): HomeVillageHeroEquipment;
    snakeBracelet(): HomeVillageHeroEquipment;
    stickHorse(): HomeVillageHeroEquipment;
    invisibilityVial(): HomeVillageHeroEquipment;
    healerPuppet(): HomeVillageHeroEquipment;
    frostFlake(): HomeVillageHeroEquipment;
    frozenArrow(): HomeVillageHeroEquipment;
    magicMirror(): HomeVillageHeroEquipment;
    actionFigure(): HomeVillageHeroEquipment;
    henchmenPuppet(): HomeVillageHeroEquipment;
    darkOrb(): HomeVillageHeroEquipment;
    metalPants(): HomeVillageHeroEquipment;
    nobleIron(): HomeVillageHeroEquipment;
    darkCrown(): HomeVillageHeroEquipment;
    meteorStaff(): HomeVillageHeroEquipment;
    eternalTome(): HomeVillageHeroEquipment;
    lifeGem(): HomeVillageHeroEquipment;
    rageGem(): HomeVillageHeroEquipment;
    healingTome(): HomeVillageHeroEquipment;
    flameBlower(): HomeVillageHeroEquipment;
    fireHeart(): HomeVillageHeroEquipment;
    fireball(): HomeVillageHeroEquipment;
    lavaloonPuppet(): HomeVillageHeroEquipment;
    heroicTorch(): HomeVillageHeroEquipment;
    royalGem(): HomeVillageHeroEquipment;
    seekingShield(): HomeVillageHeroEquipment;
    hogRiderPuppet(): HomeVillageHeroEquipment;
    hasteVial(): HomeVillageHeroEquipment;
    rocketBackpack(): HomeVillageHeroEquipment;
    rocketSpear(): HomeVillageHeroEquipment;
    /** Filter to equipment belonging to the given hero (by hero ID, e.g. `'barbarian-king'`). */
    byHero(heroId: string): HomeVillageHeroEquipment;
    /** Filter by equipment rarity (`'common'`, `'epic'`, etc.). */
    byRarity(rarity: EquipmentRarity): HomeVillageHeroEquipment;
    /** Filter to equipment that has at least one level available at or below the given Blacksmith level. */
    byBlacksmith(level: number): HomeVillageHeroEquipment;
}

/**
 * Query class for all Home Village heroes.
 * Returned by `home().heroes()`.
 */
declare class HomeVillageHeroes extends QueryBase<HomeHero> {
    constructor(data?: HomeHero[]);
    barbarianKing(): HomeVillageHeroes;
    archerQueen(): HomeVillageHeroes;
    grandWarden(): HomeVillageHeroes;
    royalChampion(): HomeVillageHeroes;
    dragonDuke(): HomeVillageHeroes;
    minionPrince(): HomeVillageHeroes;
    /** Filter to heroes that have at least one level available at or below the given Hero Hall level. */
    byHeroHall(level: number): HomeVillageHeroes;
}

/** Query class for the Helper Hut building. Returned by `home().otherBuildings().helperHut()`. */
declare class HomeVillageHelperHut extends QueryBase<HomeHelperHutBuilding> {
    constructor(data?: HomeHelperHutBuilding[]);
}
/** Query class for the Lab Assistant helper. Returned by `home().otherBuildings().helpers().labAssistant()`. */
declare class HomeVillageLabAssistant extends QueryBase<HomeLabAssistantHelper> {
    constructor(data?: HomeLabAssistantHelper[]);
}
/** Query class for the Builder's Apprentice helper. Returned by `home().otherBuildings().helpers().buildersApprentice()`. */
declare class HomeVillageBuildersApprentice extends QueryBase<HomeBuilderApprenticeHelper> {
    constructor(data?: HomeBuilderApprenticeHelper[]);
}
/** Query class for the Alchemist helper. Returned by `home().otherBuildings().helpers().alchemist()`. */
declare class HomeVillageAlchemist extends QueryBase<HomeAlchemistHelper> {
    constructor(data?: HomeAlchemistHelper[]);
}
/** Query class for the Prospector helper. Returned by `home().otherBuildings().helpers().prospector()`. */
declare class HomeVillageProspector extends QueryBase<HomeProspectorHelper> {
    constructor(data?: HomeProspectorHelper[]);
}
/**
 * Query class for all Home Village helper units (Lab Assistant, Builder's Apprentice, Alchemist, Prospector).
 * Returned by `home().otherBuildings().helpers()`.
 */
declare class HomeVillageHelpers extends QueryBase<HomeHelper<HomeHelperLevel>> {
    constructor(data?: HomeHelper<HomeHelperLevel>[]);
    labAssistant(): HomeVillageLabAssistant;
    buildersApprentice(): HomeVillageBuildersApprentice;
    alchemist(): HomeVillageAlchemist;
    prospector(): HomeVillageProspector;
    /** Filter to helpers unlocked at or before the given Town Hall level. */
    byTownHall(level: number): HomeVillageHelpers;
}
/**
 * Query class for other Home Village buildings (Bob's Hut) and the helpers sub-namespace.
 * Returned by `home().otherBuildings()`.
 */
declare class HomeVillageOtherBuildings extends QueryBase<HomeOtherBuilding> {
    constructor(data?: HomeOtherBuilding[]);
    bobsHut(): HomeVillageOtherBuildings;
    helperHut(): HomeVillageHelperHut;
    /** Returns a query over all helper units (Lab Assistant, Builder's Apprentice, Alchemist, Prospector). */
    helpers(): HomeVillageHelpers;
    /** Filter to other buildings available (count > 0) at the given Town Hall level. */
    byTownHall(level: number): HomeVillageOtherBuildings;
}

/**
 * Query class for all Home Village pets.
 * Returned by `home().pets()`.
 */
declare class HomeVillagePets extends QueryBase<HomePet> {
    constructor(data?: HomePet[]);
    lassi(): HomeVillagePets;
    electroOwl(): HomeVillagePets;
    mightyYak(): HomeVillagePets;
    unicorn(): HomeVillagePets;
    frosty(): HomeVillagePets;
    diggy(): HomeVillagePets;
    poisonLizard(): HomeVillagePets;
    phoenix(): HomeVillagePets;
    spiritFox(): HomeVillagePets;
    angryJelly(): HomeVillagePets;
    sneezy(): HomeVillagePets;
    greedyRaven(): HomeVillagePets;
    /** Filter to pets that are unlocked at or below the given Pet House level. */
    byPetHouse(level: number): HomeVillagePets;
    /** Filter to pets that have at least one level available at or before the given Town Hall level. */
    byTownHall(level: number): HomeVillagePets;
}

/** Query class for the Clan Castle building. Returned by `home().resourceBuildings().clanCastle()`. */
declare class HomeVillageClanCastle extends QueryBase<HomeClanCastle> {
    constructor(data?: HomeClanCastle[]);
}
/**
 * Query class for all Home Village resource buildings (collectors and storages).
 * Returned by `home().resourceBuildings()`.
 */
declare class HomeVillageResourceBuildings extends QueryBase<HomeResourceBuilding> {
    constructor(data?: HomeResourceBuilding[]);
    goldMine(): HomeVillageResourceBuildings;
    elixirCollector(): HomeVillageResourceBuildings;
    darkElixirDrill(): HomeVillageResourceBuildings;
    goldStorage(): HomeVillageResourceBuildings;
    elixirStorage(): HomeVillageResourceBuildings;
    darkElixirStorage(): HomeVillageResourceBuildings;
    clanCastle(): HomeVillageClanCastle;
    /** Filter to resource buildings available (count > 0) at the given Town Hall level. */
    byTownHall(level: number): HomeVillageResourceBuildings;
}

/**
 * Query class for all Home Village siege machines.
 * Returned by `home().siegeMachines()`.
 */
declare class HomeVillageSiegeMachines extends QueryBase<SiegeMachine> {
    constructor(data?: SiegeMachine[]);
    wallWrecker(): HomeVillageSiegeMachines;
    battleBlimp(): HomeVillageSiegeMachines;
    stoneSlammer(): HomeVillageSiegeMachines;
    siegeBarracks(): HomeVillageSiegeMachines;
    logLauncher(): HomeVillageSiegeMachines;
    flameFlinger(): HomeVillageSiegeMachines;
    battleDrill(): HomeVillageSiegeMachines;
    troopLauncher(): HomeVillageSiegeMachines;
    /** Filter to siege machines unlocked at or below the given Workshop level. */
    byWorkshop(level: number): HomeVillageSiegeMachines;
    /** Filter to siege machines that have at least one level available at or before the given Town Hall level. */
    byTownHall(level: number): HomeVillageSiegeMachines;
}

/**
 * Query class for all Home Village spells.
 * Returned by `home().spells()`.
 */
declare class HomeVillageSpells extends QueryBase<HomeSpell> {
    constructor(data?: HomeSpell[]);
    lightningSpell(): HomeVillageSpells;
    healingSpell(): HomeVillageSpells;
    rageSpell(): HomeVillageSpells;
    jumpSpell(): HomeVillageSpells;
    freezeSpell(): HomeVillageSpells;
    cloneSpell(): HomeVillageSpells;
    invisibilitySpell(): HomeVillageSpells;
    recallSpell(): HomeVillageSpells;
    reviveSpell(): HomeVillageSpells;
    earthquakeSpell(): HomeVillageSpells;
    hasteSpell(): HomeVillageSpells;
    poisonSpell(): HomeVillageSpells;
    totemSpell(): HomeVillageSpells;
    skeletonSpell(): HomeVillageSpells;
    batSpell(): HomeVillageSpells;
    overgrowthSpell(): HomeVillageSpells;
    iceBlockSpell(): HomeVillageSpells;
    /** Filter by spell type — `'regular'` (brewed in Spell Factory) or `'dark'` (Dark Spell Factory). */
    byType(type: 'regular' | 'dark'): HomeVillageSpells;
    /** Filter to spells that have at least one level available at or before the given Town Hall level. */
    byTownHall(level: number): HomeVillageSpells;
}

/**
 * Query class wrapping the single Town Hall entity.
 * Returned by `home().townHall()`.
 */
declare class HomeVillageTownHall extends QueryBase<TownHall> {
    constructor();
}

declare class HomeVillageTraps extends QueryBase<HomeTrap> {
    constructor(data?: HomeTrap[]);
    bomb(): HomeVillageTraps;
    springTrap(): HomeVillageTraps;
    giantBomb(): HomeVillageTraps;
    airBomb(): HomeVillageTraps;
    seekingAirMine(): HomeVillageTraps;
    skeletonTrap(): HomeVillageTraps;
    tornadoTrap(): HomeVillageTraps;
    gigaBomb(): HomeVillageTraps;
    /** Filter to a specific trap by name (case-insensitive). */
    byTrap(name: string): HomeVillageTraps;
    /** Filter to traps available (count > 0) at the given Town Hall level. */
    byTownHall(level: number): HomeVillageTraps;
    /** Filter by target type. */
    byTargetType(type: HomeTrap['targetType']): HomeVillageTraps;
}

/**
 * Query class for all Home Village troops.
 * Returned by `home().troops()`.
 */
declare class HomeVillageTroops extends QueryBase<HomeTroop> {
    constructor(data?: HomeTroop[]);
    apprenticeWarden(): HomeVillageTroops;
    barbarian(): HomeVillageTroops;
    archer(): HomeVillageTroops;
    giant(): HomeVillageTroops;
    goblin(): HomeVillageTroops;
    golem(): HomeVillageTroops;
    wallBreaker(): HomeVillageTroops;
    balloon(): HomeVillageTroops;
    wizard(): HomeVillageTroops;
    healer(): HomeVillageTroops;
    dragon(): HomeVillageTroops;
    druid(): HomeVillageTroops;
    furnace(): HomeVillageTroops;
    pekka(): HomeVillageTroops;
    babyDragon(): HomeVillageTroops;
    miner(): HomeVillageTroops;
    electroDragon(): HomeVillageTroops;
    yeti(): HomeVillageTroops;
    dragonRider(): HomeVillageTroops;
    electroTitan(): HomeVillageTroops;
    rootRider(): HomeVillageTroops;
    thrower(): HomeVillageTroops;
    meteorGolem(): HomeVillageTroops;
    minion(): HomeVillageTroops;
    hogRider(): HomeVillageTroops;
    valkyrie(): HomeVillageTroops;
    witch(): HomeVillageTroops;
    lavaHound(): HomeVillageTroops;
    bowler(): HomeVillageTroops;
    iceGolem(): HomeVillageTroops;
    headhunter(): HomeVillageTroops;
    /** Filter by troop type — `'regular'` (trained in Barracks) or `'dark'` (Dark Barracks). */
    byType(type: 'regular' | 'dark'): HomeVillageTroops;
    /** Filter to troops that have at least one level available at or before the given Town Hall level. */
    byTownHall(level: number): HomeVillageTroops;
    /** Filter to troops that have an associated Super Troop variant. */
    withSuperTroop(): HomeVillageTroops;
}

/**
 * Query class for Home Village wall data.
 * Returned by `home().walls()`.
 */
declare class HomeVillageWalls extends QueryBase<HomeWall> {
    constructor(data?: HomeWall[]);
    wall(): HomeVillageWalls;
    /** Filter to walls available (count > 0) at the given Town Hall level. */
    byTownHall(level: number): HomeVillageWalls;
}

/** Namespace for all Home Village entity queries. Use the {@link home} factory to create an instance. */
declare class HomeVillage {
    /** Returns a query over all army buildings (Army Camp, Barracks, Laboratory, etc.). */
    armyBuildings(): HomeVillageArmyBuildings;
    /** Returns a query over all stationary defenses (Cannon, Archer Tower, Mortar, etc.). */
    defenses(): HomeVillageDefenses;
    /** Returns a query over all crafted defenses (Hook Tower, Roaster, etc.). */
    craftedDefenses(): HomeVillageCraftedDefenses;
    /** Returns a query over all Guardians (Longshot, Smasher). */
    guardians(): HomeVillageGuardians;
    /** Returns a query over all hero equipment items. */
    heroEquipment(): HomeVillageHeroEquipment;
    /** Returns a query over all heroes (Barbarian King, Archer Queen, etc.). */
    heroes(): HomeVillageHeroes;
    /** Returns a query over all traps (Bomb, Spring Trap, Giant Bomb, etc.). */
    traps(): HomeVillageTraps;
    /** Returns a query over Wall data. */
    walls(): HomeVillageWalls;
    /** Returns a query wrapping the Town Hall entity. */
    townHall(): HomeVillageTownHall;
    /** Returns a query over all resource buildings (Gold Mine, Elixir Collector, storages, etc.). */
    resourceBuildings(): HomeVillageResourceBuildings;
    /** Returns a query over all spells (Lightning, Rage, Freeze, etc.). */
    spells(): HomeVillageSpells;
    /** Returns a query over all home village troops (Barbarian, Archer, Dragon, etc.). */
    troops(): HomeVillageTroops;
    /** Returns a query over all pets (Lassi, Electro Owl, Mighty Yak, etc.). */
    pets(): HomeVillagePets;
    /** Returns a query over all siege machines (Wall Wrecker, Battle Blimp, etc.). */
    siegeMachines(): HomeVillageSiegeMachines;
    /** Returns a query over other buildings (Bob's Hut) and helpers (Helper Hut, Lab Assistant, etc.). */
    otherBuildings(): HomeVillageOtherBuildings;
    /**
     * Computes the total number of upgradeable level slots available at a given Town Hall level,
     * broken down by category. Useful for progress tracking and upgrade completion ratios.
     *
     * @param thLevel - Town Hall level (1–18)
     */
    levelCountAtTownHall(thLevel: number): TownHallLevelCounts;
}
/** Creates a new {@link HomeVillage} namespace instance. Entry point for all Home Village data. */
declare function home(): HomeVillage;

declare class BuilderBaseReinforcementCamp extends QueryBase<BuilderArmyCampBuilding> {
    constructor(data?: BuilderArmyCampBuilding[]);
    /**
     * Returns a new query with each camp's `instances` filtered to those
     * unlockable at or before the given Builder Hall level.
     */
    byBuilderHall(level: number): BuilderBaseReinforcementCamp;
}
type BuilderArmyBuildingItem = BuilderArmyBuilding | BuilderBarracksBuilding | BuilderHealingHutBuilding;
declare class BuilderBaseArmyCamp extends QueryBase<BuilderArmyCampBuilding> {
    constructor(data?: BuilderArmyCampBuilding[]);
    /**
     * Returns a new query with each camp's `instances` filtered to those
     * unlockable at or before the given Builder Hall level.
     * Mirrors how `byBuilderHall` on other buildings filters by level.
     */
    byBuilderHall(level: number): BuilderBaseArmyCamp;
}
/**
 * Query class for all Builder Base army buildings.
 * Returned by `builder().armyBuildings()`.
 */
declare class BuilderBaseArmyBuildings extends QueryBase<BuilderArmyBuildingItem> {
    constructor(data?: BuilderArmyBuildingItem[]);
    builderBarracks(): BuilderBaseArmyBuildings;
    armyCamp(): BuilderBaseArmyCamp;
    starLaboratory(): BuilderBaseArmyBuildings;
    battleMachineAltar(): BuilderBaseArmyBuildings;
    battleCopterAltar(): BuilderBaseArmyBuildings;
    reinforcementCamp(): BuilderBaseReinforcementCamp;
    healingHut(): BuilderBaseArmyBuildings;
    /** Filter to army buildings available (count > 0) at the given Builder Hall level. */
    byBuilderHall(level: number): BuilderBaseArmyBuildings;
}

/**
 * Query class wrapping the single Builder Hall entity.
 * Returned by `builder().builderHall()`.
 */
declare class BuilderBaseBuilderHall extends QueryBase<BuilderHall> {
    constructor();
}

/**
 * Query class for all Builder Base stationary defenses.
 * Returned by `builder().defenses()`.
 */
declare class BuilderBaseDefenses extends QueryBase<BuilderDefense> {
    constructor(data?: BuilderDefense[]);
    cannon(): BuilderBaseDefenses;
    doubleCannon(): BuilderBaseDefenses;
    archerTower(): BuilderBaseDefenses;
    hiddenTesla(): BuilderBaseDefenses;
    firecrackers(): BuilderBaseDefenses;
    crusher(): BuilderBaseDefenses;
    guardPost(): BuilderBaseDefenses;
    airBombs(): BuilderBaseDefenses;
    multiMortar(): BuilderBaseDefenses;
    ottosOutpost(): BuilderBaseDefenses;
    roaster(): BuilderBaseDefenses;
    giantCannon(): BuilderBaseDefenses;
    megaTesla(): BuilderBaseDefenses;
    lavaLauncher(): BuilderBaseDefenses;
    xBow(): BuilderBaseDefenses;
    /** Filter to defenses available at the given Builder Hall level. */
    byBuilderHall(level: number): BuilderBaseDefenses;
    /** Filter to defenses that deal a specific damage type. */
    byDamageType(type: BuilderDefense['modes']['normal']['damageType']): BuilderBaseDefenses;
}

/**
 * Query class for all Builder Base heroes.
 * Returned by `builder().heroes()`.
 */
declare class BuilderBaseHeroes extends QueryBase<BuilderHero> {
    constructor(data?: BuilderHero[]);
    battleMachine(): BuilderBaseHeroes;
    battleCopter(): BuilderBaseHeroes;
    /** Filter to heroes that have at least one level available at or below the given Builder Hall level. */
    byBuilderHall(level: number): BuilderBaseHeroes;
}

/**
 * Query class for Builder Base leagues.
 * Returned by `builder().leagues()`.
 */
declare class BuilderBaseLeagues extends QueryBase<BuilderBaseLeague> {
    /** Filter to leagues belonging to a specific group (e.g. `"Wood"`, `"Diamond"`). */
    byGroup(group: BuilderBaseLeagueGroup): BuilderBaseLeagues;
    /**
     * Filter leagues by a partial, case-insensitive name search.
     * e.g. `byName('Iron')` returns Iron III, Iron II, Iron I.
     *      `byName('Iron I')` returns only Iron I.
     */
    byName(query: string): BuilderBaseLeagues;
    /**
     * Return the league that contains the given trophy count, or `undefined` if out of range.
     * Diamond league matches any value ≥ 6200.
     */
    atTrophies(trophies: number): BuilderBaseLeague | undefined;
}

/**
 * Query class for all Builder Base other buildings.
 * Returned by `builder().otherBuildings()`.
 */
declare class BuilderBaseOtherBuildings extends QueryBase<BuilderOtherBuilding> {
    constructor(data?: BuilderOtherBuilding[]);
    clockTower(): BuilderBaseOtherBuildings;
    /** Filter to other buildings available (count > 0) at the given Builder Hall level. */
    byBuilderHall(level: number): BuilderBaseOtherBuildings;
}

/**
 * Query class for all Builder Base resource buildings.
 * Returned by `builder().resourceBuildings()`.
 */
declare class BuilderBaseResourceBuildings extends QueryBase<BuilderResourceBuilding> {
    constructor(data?: BuilderResourceBuilding[]);
    goldMine(): BuilderBaseResourceBuildings;
    elixirCollector(): BuilderBaseResourceBuildings;
    goldStorage(): BuilderBaseResourceBuildings;
    elixirStorage(): BuilderBaseResourceBuildings;
    gemMine(): BuilderBaseResourceBuildings;
    bobControl(): BuilderBaseResourceBuildings;
    /** Filter to resource buildings available (count > 0) at the given Builder Hall level. */
    byBuilderHall(level: number): BuilderBaseResourceBuildings;
}

/**
 * Query class for all Builder Base traps.
 * Returned by `builder().traps()`.
 */
declare class BuilderBaseTraps extends QueryBase<BuilderTrap> {
    constructor(data?: BuilderTrap[]);
    pushTrap(): BuilderBaseTraps;
    springTrap(): BuilderBaseTraps;
    mine(): BuilderBaseTraps;
    megaMine(): BuilderBaseTraps;
    /** Filter to traps available (count > 0) at the given Builder Hall level. */
    byBuilderHall(level: number): BuilderBaseTraps;
}

/**
 * Query class for all Builder Base troops.
 * Returned by `builder().troops()`.
 */
declare class BuilderBaseTroops extends QueryBase<BuilderTroop> {
    constructor(data?: BuilderTroop[]);
    babyDragon(): BuilderBaseTroops;
    betaMinion(): BuilderBaseTroops;
    bomber(): BuilderBaseTroops;
    cannonCart(): BuilderBaseTroops;
    dropShip(): BuilderBaseTroops;
    electrofireWizard(): BuilderBaseTroops;
    hogGlider(): BuilderBaseTroops;
    nightWitch(): BuilderBaseTroops;
    powerPekka(): BuilderBaseTroops;
    boxerGiant(): BuilderBaseTroops;
    ragedBarbarian(): BuilderBaseTroops;
    sneakyArcher(): BuilderBaseTroops;
    /** Filter to troops with the given damage type. */
    byDamageType(type: BuilderTroop['damageType']): BuilderBaseTroops;
    /** Filter to troops that target the given target type. */
    byTargetType(type: BuilderTroop['targetType']): BuilderBaseTroops;
}

/**
 * Query class for all Builder Base wall data.
 * Returned by `builder().walls()`.
 */
declare class BuilderBaseWalls extends QueryBase<BuilderWall> {
    constructor(data?: BuilderWall[]);
    wall(): BuilderBaseWalls;
    /** Filter to walls available (count > 0) at the given Builder Hall level. */
    byBuilderHall(level: number): BuilderBaseWalls;
}

/** Namespace for all Builder Base entity queries. Use the {@link builder} factory to create an instance. */
declare class BuilderBase {
    /** Returns a query wrapping the Builder Hall entity. */
    builderHall(): BuilderBaseBuilderHall;
    /** Returns a query over all stationary defenses (Cannon, etc.). */
    defenses(): BuilderBaseDefenses;
    /** Returns a query over Builder Base traps. */
    traps(): BuilderBaseTraps;
    /** Returns a query over Builder Base wall data. */
    walls(): BuilderBaseWalls;
    /** Returns a query over Builder Base resource buildings. */
    resourceBuildings(): BuilderBaseResourceBuildings;
    /** Returns a query over Builder Base army buildings. */
    armyBuildings(): BuilderBaseArmyBuildings;
    /** Returns a query over other Builder Base buildings (Clock Tower, etc.). */
    otherBuildings(): BuilderBaseOtherBuildings;
    /** Returns a query over Builder Base troops. */
    troops(): BuilderBaseTroops;
    /** Returns a query over Builder Base heroes. */
    heroes(): BuilderBaseHeroes;
    /** Returns a query over all Builder Base leagues. */
    leagues(): BuilderBaseLeagues;
    /**
     * Computes the total number of upgradeable level slots available at a given Builder Hall level,
     * broken down by category. Useful for progress tracking and upgrade completion ratios.
     *
     * @param bhLevel - Builder Hall level (1–10)
     */
    levelCountAtBuilderHall(bhLevel: number): BuilderHallLevelCounts;
}
/** Creates a new {@link BuilderBase} namespace instance. Entry point for all Builder Base data. */
declare function builder(): BuilderBase;

/**
 * Query class for all Clan Capital barracks buildings.
 * Returned by `clanCapital().armyBuildings().barracks()`.
 */
declare class ClanCapitalBarracks extends QueryBase<ClanCapitalBarracksBuilding> {
    constructor(data?: ClanCapitalBarracksBuilding[]);
    superBarbarianBarracks(): ClanCapitalBarracks;
    sneakyArcherBarracks(): ClanCapitalBarracks;
    superGiantBarracks(): ClanCapitalBarracks;
    battleRamBarracks(): ClanCapitalBarracks;
    minionBarracks(): ClanCapitalBarracks;
    superWizardBarracks(): ClanCapitalBarracks;
    rocketBalloonBarracks(): ClanCapitalBarracks;
    skeletonBarrelBarracks(): ClanCapitalBarracks;
    flyingFortressYard(): ClanCapitalBarracks;
    raidCartBarracks(): ClanCapitalBarracks;
    powerPekkaBarracks(): ClanCapitalBarracks;
    hogRaiderBarracks(): ClanCapitalBarracks;
    superDragonBarracks(): ClanCapitalBarracks;
    mountainGolemQuarry(): ClanCapitalBarracks;
    infernoDragonBarracks(): ClanCapitalBarracks;
    superMinerBarracks(): ClanCapitalBarracks;
    megaSparkyWorkshop(): ClanCapitalBarracks;
    /** Filter to barracks available in the given district. */
    byDistrict(district: string): ClanCapitalBarracks;
}

/**
 * Query class for all Clan Capital spell factory buildings.
 * Returned by `clanCapital().armyBuildings().spellFactories()`.
 */
declare class ClanCapitalSpellFactories extends QueryBase<ClanCapitalSpellFactory> {
    constructor(data?: ClanCapitalSpellFactory[]);
    healSpellFactory(): ClanCapitalSpellFactories;
    jumpSpellFactory(): ClanCapitalSpellFactories;
    lightningSpellFactory(): ClanCapitalSpellFactories;
    frostSpellFactory(): ClanCapitalSpellFactories;
    rageSpellFactory(): ClanCapitalSpellFactories;
    graveyardSpellFactory(): ClanCapitalSpellFactories;
    endlessHasteSpellFactory(): ClanCapitalSpellFactories;
    /** Filter to spell factories available in the given district. */
    byDistrict(district: string): ClanCapitalSpellFactories;
}

type ClanCapitalArmyBuildingItem = ClanCapitalArmyBuilding | ClanCapitalSpellStorageBuilding | ClanCapitalBarracksBuilding;

/**
 * Query class for all Clan Capital army buildings.
 * Returned by `clanCapital().armyBuildings()`.
 */
declare class ClanCapitalArmyBuildings extends QueryBase<ClanCapitalArmyBuildingItem> {
    constructor(data?: ClanCapitalArmyBuildingItem[]);
    armyCamp(): ClanCapitalArmyBuildings;
    spellStorage(): ClanCapitalArmyBuildings;
    barracks(): ClanCapitalBarracks;
    spellFactories(): ClanCapitalSpellFactories;
    /** Filter to army buildings available in the given district. */
    byDistrict(district: string): ClanCapitalArmyBuildings;
}

/**
 * Query class for all Clan Capital stationary defenses.
 * Returned by `clanCapital().defenses()`.
 */
declare class ClanCapitalDefenses extends QueryBase<ClanCapitalDefense> {
    constructor(data?: ClanCapitalDefense[]);
    cannon(): ClanCapitalDefenses;
    spearThrower(): ClanCapitalDefenses;
    airDefense(): ClanCapitalDefenses;
    multiCannon(): ClanCapitalDefenses;
    bombTower(): ClanCapitalDefenses;
    multiMortar(): ClanCapitalDefenses;
    airBombs(): ClanCapitalDefenses;
    blastBow(): ClanCapitalDefenses;
    raidCartPost(): ClanCapitalDefenses;
    rapidRockets(): ClanCapitalDefenses;
    crusher(): ClanCapitalDefenses;
    giantCannon(): ClanCapitalDefenses;
    goblinThrower(): ClanCapitalDefenses;
    hiddenMegaTesla(): ClanCapitalDefenses;
    infernoTower(): ClanCapitalDefenses;
    miniMinionHive(): ClanCapitalDefenses;
    superDragonPost(): ClanCapitalDefenses;
    superGiantPost(): ClanCapitalDefenses;
    superWizardTower(): ClanCapitalDefenses;
    rocketArtillery(): ClanCapitalDefenses;
    reflector(): ClanCapitalDefenses;
    /** Filter to defenses that target the given unit type. */
    byTargetType(type: 'ground' | 'air' | 'both'): ClanCapitalDefenses;
    /** Filter to defenses that have at least one level available at or below the given Capital Hall level. */
    byCapitalHall(level: number): ClanCapitalDefenses;
}

/** Query class for Clan Capital Forge data. Returned by `clanCapital().forge()`. */
declare class ClanCapitalForge {
    /** Capital Gold obtained from the daily forge, indexed by Town Hall level (TH6–18). */
    dailyForge(): ClanCapitalDailyForgeEntry[];
    /** Capital Gold obtained from the daily forge at a specific Town Hall level. */
    dailyForgeAtTownHall(townHallLevel: number): ClanCapitalDailyForgeEntry | undefined;
    /** Time required to complete one Auto Forge or Forge cycle. */
    craftingTime(): BuildTime;
    /**
     * Number of manual Forge slots available outside the Daily and Auto Forge,
     * each entry giving the TH level required to unlock that many slots.
     */
    availableForges(): ClanCapitalAvailableForgeEntry[];
    /** Number of manual Forge slots available at a specific Town Hall level. */
    availableForgesAtTownHall(townHallLevel: number): number;
    /**
     * Auto Forge resource conversion rates (home village + builder base).
     * Gold Pass discount prices are included for home and builder entries.
     */
    autoForge(): ClanCapitalForgeRates;
    /** Auto Forge conversion rate for a specific Town Hall level. */
    autoForgeAtTownHall(townHallLevel: number): ClanCapitalHomeForgeEntry | undefined;
    /** Auto Forge conversion rate for a specific Builder Hall level. */
    autoForgeAtBuilderHall(builderHallLevel: number): ClanCapitalBuilderForgeEntry | undefined;
    /**
     * Forge resource conversion rates (home village + builder base).
     * Higher cost than Auto Forge; no Gold Pass discount.
     */
    forgeRates(): ClanCapitalForgeRates;
    /** Forge conversion rate for a specific Town Hall level. */
    forgeAtTownHall(townHallLevel: number): ClanCapitalHomeForgeEntry | undefined;
    /** Forge conversion rate for a specific Builder Hall level. */
    forgeAtBuilderHall(builderHallLevel: number): ClanCapitalBuilderForgeEntry | undefined;
}

/**
 * Query class wrapping the single Capital Hall entity.
 * Returned by `clanCapital().capitalHall()`.
 */
declare class ClanCapitalCapitalHall extends QueryBase<CapitalHall> {
    constructor();
}

/**
 * Query class wrapping the single District Hall entity.
 * Returned by `clanCapital().districtHall()`.
 */
declare class ClanCapitalDistrictHall extends QueryBase<DistrictHall> {
    constructor();
}

/**
 * Query class for Clan Capital Raid Weekend leagues.
 * Returned by `clanCapital().leagues()`.
 */
declare class ClanCapitalLeagues extends QueryBase<ClanCapitalLeague> {
    /** Filter to leagues belonging to a specific group (e.g. `"Gold"`, `"Master"`). */
    byGroup(group: ClanCapitalLeagueGroup): ClanCapitalLeagues;
    /**
     * Return the league that contains the given trophy count, or `undefined` if out of range.
     * Legend league matches any value ≥ 5000.
     */
    atTrophies(trophies: number): ClanCapitalLeague | undefined;
}

/**
 * Query class for all Clan Capital house buildings.
 * Returned by `clanCapital().other().houses()`.
 */
declare class ClanCapitalHouses extends QueryBase<ClanCapitalHouse> {
    constructor(data?: ClanCapitalHouse[]);
    smallCabin(): ClanCapitalHouses;
    thatchedHut(): ClanCapitalHouses;
    smallHut(): ClanCapitalHouses;
    woodenHouse(): ClanCapitalHouses;
    woodenCabin(): ClanCapitalHouses;
    slantedHouse(): ClanCapitalHouses;
    goblinOutpost(): ClanCapitalHouses;
    goblinHut(): ClanCapitalHouses;
    goblinHall(): ClanCapitalHouses;
    /** Filter to houses available in the given district. */
    byDistrict(district: string): ClanCapitalHouses;
}

/**
 * Query class for all Clan Capital other buildings.
 * Returned by `clanCapital().other()`.
 */
declare class ClanCapitalOther {
    houses(): ClanCapitalHouses;
}

/**
 * Query class for all Clan Capital spells.
 * Returned by `clanCapital().spells()`.
 */
declare class ClanCapitalSpells extends QueryBase<ClanCapitalSpell> {
    constructor(data?: ClanCapitalSpell[]);
    healingSpell(): ClanCapitalSpells;
    jumpSpell(): ClanCapitalSpells;
    lightningSpell(): ClanCapitalSpells;
    frostSpell(): ClanCapitalSpells;
    graveyardSpell(): ClanCapitalSpells;
    endlessHasteSpell(): ClanCapitalSpells;
    rageSpell(): ClanCapitalSpells;
}

/**
 * Query class for all Clan Capital traps.
 * Returned by `clanCapital().traps()`.
 */
declare class ClanCapitalTraps extends QueryBase<ClanCapitalTrap> {
    constructor(data?: ClanCapitalTrap[]);
    mine(): ClanCapitalTraps;
    megaMine(): ClanCapitalTraps;
    logTrap(): ClanCapitalTraps;
    zapTrap(): ClanCapitalTraps;
    spearTrap(): ClanCapitalTraps;
    /** Filter to traps targeting the given type. */
    byTargetType(type: 'ground' | 'air' | 'both'): ClanCapitalTraps;
    /** Filter to traps available in the given district. */
    byDistrict(district: string): ClanCapitalTraps;
}

/**
 * Query class for all Clan Capital troops.
 * Returned by `clanCapital().troops()`.
 */
declare class ClanCapitalTroops extends QueryBase<ClanCapitalTroop> {
    constructor(data?: ClanCapitalTroop[]);
    superBarbarian(): ClanCapitalTroops;
    sneakyArcher(): ClanCapitalTroops;
    superGiant(): ClanCapitalTroops;
    battleRam(): ClanCapitalTroops;
    minionHorde(): ClanCapitalTroops;
    superWizard(): ClanCapitalTroops;
    rocketBalloon(): ClanCapitalTroops;
    skeletonBarrels(): ClanCapitalTroops;
    flyingFortress(): ClanCapitalTroops;
    raidCart(): ClanCapitalTroops;
    powerPekka(): ClanCapitalTroops;
    hogRaiders(): ClanCapitalTroops;
    infernoDragon(): ClanCapitalTroops;
    megaSparky(): ClanCapitalTroops;
    mountainGolem(): ClanCapitalTroops;
    superDragon(): ClanCapitalTroops;
    superMiner(): ClanCapitalTroops;
}

/**
 * Query class for Clan Capital wall data.
 * Returned by `clanCapital().walls()`.
 */
declare class ClanCapitalWalls extends QueryBase<ClanCapitalWall> {
    constructor(data?: ClanCapitalWall[]);
    wall(): ClanCapitalWalls;
    /** Filter to walls available (count > 0) at the given Capital Hall level. */
    byCapitalHall(level: number): ClanCapitalWalls;
}

/** Namespace for all Clan Capital entity queries. Use the {@link clanCapital} factory to create an instance. */
declare class ClanCapital {
    capitalHall(): ClanCapitalCapitalHall;
    districtHall(): ClanCapitalDistrictHall;
    /** Returns a query over all Clan Capital stationary defenses. */
    defenses(): ClanCapitalDefenses;
    /** Returns a query over all Clan Capital army buildings. */
    armyBuildings(): ClanCapitalArmyBuildings;
    /** Returns a query over Clan Capital other buildings (houses, etc.). */
    other(): ClanCapitalOther;
    /** Returns a query over all Clan Capital spells. */
    spells(): ClanCapitalSpells;
    /** Returns a query over all Clan Capital troops. */
    troops(): ClanCapitalTroops;
    /** Returns a query over all Clan Capital traps. */
    traps(): ClanCapitalTraps;
    /** Returns a query over Clan Capital wall data. */
    walls(): ClanCapitalWalls;
    /** Returns a query over all Clan Capital Raid Weekend leagues. */
    leagues(): ClanCapitalLeagues;
    /** Returns Forge data — daily forge, auto forge, forge rates, available slots, and crafting time. */
    forge(): ClanCapitalForge;
    /**
     * Computes the total number of upgradeable level slots available at a given Capital Hall level,
     * broken down by district/zone (structures + walls) and top-level totals for troops and spells.
     *
     * @param capitalHallLevel - Capital Hall level (1–10)
     */
    levelCountAtClanCapital(capitalHallLevel: number): ClanCapitalLevelCounts;
}
/** Creates a new {@link ClanCapital} namespace instance. Entry point for all Clan Capital data. */
declare function clanCapital(): ClanCapital;

/** Query class for boost calculations (time and cost). Returned by `calculators().boost()`. */
declare class BuildBoostCalculator {
    /** Apply a Builder Boost to reduce build time by the given tier percentage (10, 15, or 20). */
    builderBoost(time: BuildTime, tier: BoostTier): BuildTime;
    /** Apply a Research Boost to reduce research time by the given tier percentage (10, 15, or 20). */
    researchBoost(time: BuildTime, tier: BoostTier): BuildTime;
    /**
     * Apply a Builder Boost to reduce a build cost (Gold, Elixir, or Dark Elixir)
     * by the given tier percentage (10, 15, or 20). Returns the reduced cost, floored.
     */
    builderBoostCost(cost: number, resource: BuildCostResource, tier: BoostTier): number;
    /**
     * Apply a Research Boost to reduce a research cost (Gold, Elixir, or Dark Elixir)
     * by the given tier percentage (10, 15, or 20). Returns the reduced cost, floored.
     */
    researchBoostCost(cost: number, resource: BuildCostResource, tier: BoostTier): number;
}

/** Query class for Clock Tower time calculations. Returned by `calculators().clockTower()`. */
declare class ClockTowerCalculator {
    /**
     * Apply one full Clock Tower activation at the given level.
     * Reduces remaining build time by the `timeGainedMinutes` for that level.
     * Result is clamped to zero.
     */
    boost(time: BuildTime, level: ClockTowerLevel): BuildTime;
    /**
     * Apply a Clock Tower Potion (fixed 30-minute run) at the given level.
     * Time saved = floor(timeGainedMinutes × 30 / boostDurationMinutes).
     * Result is clamped to zero.
     */
    potion(time: BuildTime, level: ClockTowerLevel): BuildTime;
}

declare class GemsCalculator {
    cost(time: BuildTime): number;
}

/**
 * Query class for Helper Hut helper calculations.
 * Returned by `calculators().helpers()`.
 */
declare class HelpersCalculator {
    /**
     * Lab Assistant — reduces remaining research time by the helper's `workRate` at the given level
     * (workRate = hours of lab progress completed in 60 min of real time).
     */
    labAssistant(time: BuildTime, level: LabAssistantLevel): BuildTime;
    /**
     * Builder's Apprentice — reduces remaining build time by the helper's `workRate` at the given level.
     * Note: does not boost the Apprentice's own build queue.
     */
    buildersApprentice(time: BuildTime, level: BuildersApprenticeLevel): BuildTime;
    /**
     * Alchemist — converts Gold or Elixir to Dark Elixir at a rate of 150:1,
     * with a bonus percentage applied on top at the given level.
     * Input is capped at the level's `goldElixirConversionMax`.
     */
    alchemist(goldOrElixir: number, level: AlchemistLevel): AlchemistResult;
    /**
     * Prospector — returns the maximum daily ore conversion amounts at the given level.
     */
    prospector(level: ProspectorLevel): ProspectorResult;
}

/** Query class for time-reduction potion and snack calculations. Returned by `calculators().potions()`. */
declare class PotionsCalculator {
    /** Builder Potion: 10× builder speed for 60 min → reduces remaining build time by 10 hours. */
    builderPotion(time: BuildTime): BuildTime;
    /** Research Potion: 24× lab speed for 60 min → reduces remaining research time by 24 hours. */
    researchPotion(time: BuildTime): BuildTime;
    /** Pet Potion: 24× pet house speed for 60 min → reduces remaining pet upgrade time by 24 hours. */
    petPotion(time: BuildTime): BuildTime;
    /** Builder Bite: 2× builder speed for 60 min → reduces remaining build time by 2 hours. */
    builderBite(time: BuildTime): BuildTime;
    /** Study Soup: 4× lab speed for 60 min → reduces remaining research time by 4 hours. */
    studySoup(time: BuildTime): BuildTime;
}

/**
 * Top-level calculator namespace.
 * Returned by `calculators()`.
 */
declare class Calculators {
    /** Build time and cost boost calculations (Builder Boost, Research Boost). */
    boost(): BuildBoostCalculator;
    /** Gem cost calculator — converts build/research time into gem cost. */
    gems(): GemsCalculator;
    /** Helper Hut helper calculations (Lab Assistant, Builder's Apprentice, Alchemist, Prospector). */
    helpers(): HelpersCalculator;
    /** Time-reduction potion and snack calculations. */
    potions(): PotionsCalculator;
    /** Clock Tower boost and potion calculations (time saved per level). */
    clockTower(): ClockTowerCalculator;
}
/** Returns the Calculators namespace. */
declare function calculators(): Calculators;

/**
 * Query class for all magic item books.
 * Returned by `magicItems().books()`.
 */
declare class MagicItemBooks extends QueryBase<MagicBook> {
    bookOfFighting(): MagicItemBooks;
    bookOfBuilding(): MagicItemBooks;
    bookOfSpells(): MagicItemBooks;
    bookOfHeroes(): MagicItemBooks;
    bookOfEverything(): MagicItemBooks;
    /** Filter to books that apply to the given upgrade type. */
    byAppliesTo(appliesTo: 'troops' | 'buildings' | 'spells' | 'heroes-and-pets' | 'any'): MagicItemBooks;
}

/**
 * Query class for all magic item hammers.
 * Returned by `magicItems().hammers()`.
 */
declare class MagicItemHammers extends QueryBase<MagicHammer> {
    hammerOfFighting(): MagicItemHammers;
    hammerOfBuilding(): MagicItemHammers;
    hammerOfSpells(): MagicItemHammers;
    hammerOfHeroes(): MagicItemHammers;
    /** Filter to hammers that apply to the given upgrade type. */
    byAppliesTo(appliesTo: 'troops' | 'buildings' | 'spells' | 'heroes-and-pets'): MagicItemHammers;
}

/**
 * Query class for all magic item potions.
 * Returned by `magicItems().potions()`.
 */
declare class MagicItemPotions extends QueryBase<MagicPotion> {
    builderPotion(): MagicItemPotions;
    researchPotion(): MagicItemPotions;
    petPotion(): MagicItemPotions;
    heroPotion(): MagicItemPotions;
    powerPotion(): MagicItemPotions;
    resourcePotion(): MagicItemPotions;
    clockTowerPotion(): MagicItemPotions;
    superPotion(): MagicItemPotions;
    /** Filter to potions whose effect type matches the given value. */
    byEffectType(type: MagicPotion['effect']['type']): MagicItemPotions;
}

/**
 * Query class for all magic item snacks.
 * Returned by `magicItems().snacks()`.
 */
declare class MagicItemSnacks extends QueryBase<MagicSnack> {
    builderBite(): MagicItemSnacks;
    studySoup(): MagicItemSnacks;
    mightyMorsel(): MagicItemSnacks;
    powerPancakes(): MagicItemSnacks;
    clanCastleCake(): MagicItemSnacks;
    /** Filter to snacks whose effect type matches the given value. */
    byEffectType(type: MagicSnack['effect']['type']): MagicItemSnacks;
}

/**
 * Query class for all magic item utilities.
 * Returned by `magicItems().utilities()`.
 */
declare class MagicItemUtilities extends QueryBase<MagicUtility> {
    shovelOfObstacles(): MagicItemUtilities;
    builderStarJar(): MagicItemUtilities;
    wallRing(): MagicItemUtilities;
    /** Filter to utilities whose effect type matches the given value. */
    byEffectType(type: MagicUtility['effect']['type']): MagicItemUtilities;
}

/**
 * Top-level magic items namespace.
 * Returned by `magicItems()`.
 */
declare class MagicItems {
    /** All magic item snacks. */
    snacks(): MagicItemSnacks;
    /** All magic item potions. */
    potions(): MagicItemPotions;
    /** All magic item books. */
    books(): MagicItemBooks;
    /** All magic item hammers. */
    hammers(): MagicItemHammers;
    /** All magic item utilities (Wall Ring, Shovel of Obstacles, Builder Star Jar). */
    utilities(): MagicItemUtilities;
}
/** Returns the MagicItems namespace. */
declare function magicItems(): MagicItems;

/**
 * Query class for all Season Pass challenge task types.
 * Returned by `seasonPass().challenges()`.
 */
declare class SeasonPassChallenges extends QueryBase<SeasonPassChallenge> {
    buildingUpgrade(): SeasonPassChallenges;
    troopUpgrade(): SeasonPassChallenges;
    heroPetUpgrade(): SeasonPassChallenges;
    donateReinforcements(): SeasonPassChallenges;
    starBonus(): SeasonPassChallenges;
    winBattle(): SeasonPassChallenges;
    destroyTownHall(): SeasonPassChallenges;
    requestReinforcements(): SeasonPassChallenges;
}

/**
 * Top-level Season Pass namespace.
 * Returned by `seasonPass()`.
 */
declare class SeasonPass {
    /** All Season Pass challenge task types. */
    challenges(): SeasonPassChallenges;
}
/** Returns the SeasonPass namespace. */
declare function seasonPass(): SeasonPass;

/**
 * Query class for clan level progression data.
 * Returned by `clan().levels()`.
 */
declare class ClanLevels {
    private readonly data;
    constructor(data: ClanLevel[]);
    /** Return all clan levels. */
    get(): ClanLevel[];
    /** Return the number of clan levels. */
    count(): number;
    /** Return the entry for a specific clan level number, or `undefined` if not found. */
    atLevel(level: number): ClanLevel | undefined;
    /** Filter to levels that display a specific badge. */
    byBadge(badge: ClanBadge): ClanLevels;
}
/**
 * Query class for clan label types.
 * Returned by `clan().labels()`.
 */
declare class ClanLabels extends QueryBase<ClanLabel> {
    byId(id: string): ClanLabels;
}

/**
 * Query class for clan war data — loot, bonuses, and ore.
 * Returned by `clan().war()`.
 */
declare class ClanWar {
    /** Maximum available loot in an enemy war base, indexed by Town Hall level (TH3–18). */
    maxWarBaseLoot(): WarBaseLootEntry[];
    /** Maximum available ore in an enemy war base, indexed by Town Hall level (TH8–18). */
    maxWarBaseOre(): WarBaseOreEntry[];
    /** War bonus tiers, grouped by clan level range. */
    warBonus(): WarBonusTier[];
    /**
     * Return the war bonus tier that applies to the given clan level.
     * Returns `undefined` if the clan level is below 1.
     */
    bonusTierForClanLevel(clanLevel: number): WarBonusTier | undefined;
}

/**
 * Top-level Clan namespace.
 * Returned by `clan()`.
 */
declare class Clan {
    /** Clan level progression — XP requirements, badges, and perks per level. */
    levels(): ClanLevels;
    /** Clan labels that can be displayed on a clan's profile. */
    labels(): ClanLabels;
    /** Clan war data — max base loot, war bonuses, and ore per Town Hall level. */
    war(): ClanWar;
}
/** Returns the Clan namespace. */
declare function clan(): Clan;

/**
 * Query class for all in-game achievements.
 * Returned by `achievements()`.
 */
declare class Achievements extends QueryBase<Achievement> {
    /** Filter to achievements for a specific base (home, builder, clan-capital). */
    byBase(base: AchievementBase): Achievements;
    /** Filter to achievements that involve the given data description (case-insensitive substring). */
    byDataInvolved(keyword: string): Achievements;
    /** Filter to achievements that have exactly the given number of tiers. */
    byTierCount(count: number): Achievements;
}
/** Returns the Achievements query class. */
declare function achievements(): Achievements;

/**
 * Query class for ranked battle leagues.
 * Returned by `rankedBattles().leagues()`.
 */
declare class RankedBattlesLeagues extends QueryBase<RankedBattleLeague> {
    /** Filter to leagues belonging to a specific group (e.g. `"Skeleton"`, `"Electro"`). */
    byGroup(group: LeagueGroup): RankedBattlesLeagues;
    /**
     * Filter leagues by a partial, case-insensitive name search.
     * e.g. `byName('Dragon')` returns Dragon 28, 29, 30.
     *      `byName('Dragon 30')` returns only Dragon 30.
     */
    byName(query: string): RankedBattlesLeagues;
    /** Filter to leagues that have a difficulty modifier (Electro 32, 33, Legend). */
    withDifficultyModifier(): RankedBattlesLeagues;
}
/**
 * Query class for all ranked battles data — leagues, floors, difficulty modifiers, and loot tables.
 * Returned by `rankedBattles()`.
 */
declare class RankedBattles {
    /** All ranked leagues (Unranked, Skeleton 1–33, Legend). */
    leagues(): RankedBattlesLeagues;
    /**
     * League floor entries — the minimum league each Town Hall level cannot be demoted below.
     * Covers TH7–18.
     */
    leagueFloor(): LeagueFloorEntry[];
    /**
     * Return the league floor for a specific Town Hall level, or `undefined` if not found.
     * Town Halls below 7 cannot participate in ranked battles.
     */
    floorForTownHall(townHallLevel: number): LeagueFloorEntry | undefined;
    /**
     * Difficulty modifiers for the top three leagues (Electro 32, Electro 33, Legend).
     * These apply defense and hero stat boosts/penalties during attacks.
     */
    difficultyModifiers(): DifficultyModifier[];
    /**
     * Loot data for a specific Town Hall level (7–18).
     * Each entry contains max available loot, league bonus, and star bonus for one league.
     * Returns an empty array for Town Hall levels without data.
     *
     * Note: TH8 data starts from Skeleton 2 (Skeleton 1 was not available in the source).
     */
    loot(townHallLevel: number): RankedBattleLootEntry[];
    /**
     * League bonus and star bonus data for Town Halls below 7 (TH2–6).
     * These Town Halls cannot participate in ranked leagues but still receive Gold/Elixir bonuses.
     */
    lowerThBonuses(): LowerThBonus[];
}
/** Returns the RankedBattles namespace. */
declare function rankedBattles(): RankedBattles;

export { type Achievement, type AchievementBase, type AchievementTier, Achievements, type AlchemistLevel, type AlchemistResult, type ArmyBuildingLevel, type BarracksBuildingLevel, type Base, type BatStats, type BlacksmithMaxEquipmentLevel, type BlacksmithOreCapacity, type BoostTier, BuildBoostCalculator, type BuildCostResource, type BuildTime, type BuilderArmyBuilding, type BuilderArmyBuildingLevel, type BuilderArmyCampBuilding, type BuilderArmyCampInstance, type BuilderArmyCampLevel, type BuilderBarracksBuilding, type BuilderBarracksBuildingLevel, BuilderBase, BuilderBaseArmyBuildings, type BuilderBaseBattleResult, BuilderBaseBuilderHall, BuilderBaseDefenses, BuilderBaseHeroes, type BuilderBaseLeague, type BuilderBaseLeagueGroup, BuilderBaseLeagues, BuilderBaseOtherBuildings, BuilderBaseResourceBuildings, type BuilderBaseStarBonus, BuilderBaseTraps, BuilderBaseTroops, BuilderBaseWalls, type BuilderClockTowerBuilding, type BuilderClockTowerBuildingLevel, type BuilderDefense, type BuilderDefenseLevel, type BuilderHall, type BuilderHallAvailability, type BuilderHallLevel, type BuilderHallLevelCounts, type BuilderHealingHutBuilding, type BuilderHealingHutBuildingLevel, type BuilderHero, type BuilderHeroAbility, type BuilderHeroAbilityChargeLevel, type BuilderHeroAbilityLevel, type BuilderHeroLevel, type BuilderMode, type BuilderOtherBuilding, type BuilderResearchBuilding, type BuilderResearchBuildingLevel, type BuilderResourceBuilding, type BuilderResourceBuildingLevel, type BuilderStats, type BuilderTrap, type BuilderTrapLevel, type BuilderTroop, type BuilderTroopLevel, type BuilderWall, type BuilderWallLevel, type BuildersApprenticeLevel, type Building, type BuildingLevel, type BurstDefenseMode, Calculators, type CapitalHall, type CapitalHallAvailability, type CapitalHallLevel, type CapitalHallWeaponMode, type Category, Clan, type ClanBadge, ClanCapital, type ClanCapitalArmyBuilding, type ClanCapitalArmyBuildingLevel, ClanCapitalArmyBuildings, type ClanCapitalAvailableForgeEntry, ClanCapitalBarracks, type ClanCapitalBarracksBuilding, type ClanCapitalBarracksBuildingLevel, type ClanCapitalBuilderForgeEntry, ClanCapitalCapitalHall, type ClanCapitalDailyForgeEntry, type ClanCapitalDefense, type ClanCapitalDefenseLevel, ClanCapitalDefenses, type ClanCapitalDistrict, type ClanCapitalDistrictCounts, ClanCapitalDistrictHall, ClanCapitalForge, type ClanCapitalForgeData, type ClanCapitalForgeRates, type ClanCapitalHomeForgeEntry, type ClanCapitalHouse, type ClanCapitalHouseLevel, ClanCapitalHouses, type ClanCapitalLeague, type ClanCapitalLeagueGroup, ClanCapitalLeagues, type ClanCapitalLevelCounts, ClanCapitalOther, type ClanCapitalResourceBuilding, type ClanCapitalResourceBuildingLevel, type ClanCapitalSkeletonStats, type ClanCapitalSpell, ClanCapitalSpellFactories, type ClanCapitalSpellFactory, type ClanCapitalSpellFactoryLevel, type ClanCapitalSpellLevel, type ClanCapitalSpellStorageBuilding, type ClanCapitalSpellStorageBuildingLevel, ClanCapitalSpells, type ClanCapitalTrap, type ClanCapitalTrapLevel, ClanCapitalTraps, type ClanCapitalTroop, type ClanCapitalTroopLevel, type ClanCapitalTroopStats, type ClanCapitalTroopSubUnit, type ClanCapitalTroopSubUnitLevel, ClanCapitalTroops, type ClanCapitalWall, type ClanCapitalWallLevel, ClanCapitalWalls, type ClanCastleEffect, type ClanCastleLevel, type ClanDonationLimit, type ClanLabel, ClanLabels, type ClanLevel, type ClanLevelPerks, ClanLevels, ClanWar, type ClockTowerBoostEffect, ClockTowerCalculator, type ClockTowerLevel, type CombatBoostEffect, type CraftedDefense, type CraftedDefenseImageEntry, type CraftedDefenseModule, type CraftedDefenseModuleUpgrade, type DefenseMode, type DefenseModeStats, type DifficultyModifier, type DistrictAvailability, type DistrictHall, type DistrictHallAvailability, type DistrictHallCapitalHallRequired, type DistrictHallLevel, type DonationCost, type EarthquakeSpellMode, type EquipmentRarity, type GearUp, GemsCalculator, type Guardian, type GuardianLevel, type GuardianMode, HelpersCalculator, type HeroBannerBuilding, type HeroBannerImages, type HeroEquipment, type HeroEquipmentLevel, type HeroLevelCaps, type HomeAlchemistHelper, type HomeAlchemistHelperLevel, type HomeArmyBuilding, type HomeArmyBuildingLevel, type HomeBarracksBuilding, type HomeBarracksBuildingLevel, type HomeBlacksmithBuilding, type HomeBlacksmithBuildingLevel, type HomeBuilderApprenticeHelper, type HomeClanCastle, type HomeClanCastleLevel, type HomeDefense, type HomeDefenseLevel, type HomeHelper, type HomeHelperHutBuilding, type HomeHelperHutBuildingLevel, type HomeHelperLevel, type HomeHero, type HomeHeroHallBuilding, type HomeHeroHallBuildingLevel, type HomeHeroLevel, type HomeLabAssistantHelper, type HomeOtherBuilding, type HomePet, type HomePetHouseBuilding, type HomePetHouseBuildingLevel, type HomePetLevel, type HomeProspectorHelper, type HomeProspectorHelperLevel, type HomeResearchBuilding, type HomeResearchBuildingLevel, type HomeResourceBuilding, type HomeResourceBuildingLevel, type HomeSpell, type HomeSpellFactoryBuilding, type HomeSpellFactoryBuildingLevel, type HomeSpellLevel, type HomeSuperTroop, type HomeSuperTroopLevel, type HomeTrap, type HomeTroop, type HomeTroopLevel, type HomeTroopLevelStats, HomeVillage, HomeVillageAlchemist, HomeVillageArmyBuildings, HomeVillageBarracks, HomeVillageBlacksmith, HomeVillageBuildersApprentice, HomeVillageClanCastle, HomeVillageCraftedDefenses, HomeVillageDarkBarracks, HomeVillageDarkSpellFactory, HomeVillageDefenses, HomeVillageGuardians, HomeVillageHelperHut, HomeVillageHelpers, HomeVillageHeroBanner, HomeVillageHeroEquipment, HomeVillageHeroHall, HomeVillageHeroes, HomeVillageLabAssistant, HomeVillageLaboratory, HomeVillageOtherBuildings, HomeVillagePetHouse, HomeVillagePets, HomeVillageProspector, HomeVillageResourceBuildings, HomeVillageSiegeMachines, HomeVillageSpellFactory, HomeVillageSpells, HomeVillageTownHall, HomeVillageTraps, HomeVillageTroops, HomeVillageWalls, HomeVillageWorkshop, type HomeWall, type HomeWorkRateHelperLevel, type HomeWorkshopBuilding, type HomeWorkshopBuildingLevel, type InstantCompleteEffect, type InstantUpgradeEffect, type InvisibilitySpellMode, type LabAssistantLevel, type LeagueFloorEntry, type LeagueGroup, type LongshotGuardian, type LowerThBonus, type MagicBook, type MagicHammer, type MagicItem, MagicItemBooks, type MagicItemEffect, MagicItemHammers, MagicItemPotions, MagicItemSnacks, type MagicItemType, MagicItemUtilities, MagicItems, type MagicPotion, type MagicSnack, type MagicUtility, type ObstacleMoveEffect, type OtherBuilding, type OtherBuildingLevel, type PoisonSpellMode, PotionsCalculator, type ProspectorLevel, type ProspectorResult, type RageSpellMode, type RankedBattleLeague, type RankedBattleLootAmount, type RankedBattleLootEntry, RankedBattles, RankedBattlesLeagues, type ResearchBuildingLevel, type ResourceBuildingLevel, type ResourceCollectorBoostEffect, type ResourceType, SeasonPass, type SeasonPassChallenge, SeasonPassChallenges, type SiegeMachine, type SiegeMachineLevel, type SkeletonStats, type SmasherGuardian, type SpellFactoryBuildingLevel, type SpellTowerMode, type StarBonusOre, type StarBonusResetEffect, type SuperTroopEffect, type TimeReductionEffect, type TownHall, type TownHallAvailability, type TownHallLevel, type TownHallLevelCounts, type TownHallStorageCapacity, type TownHallWeapon, type TownHallWeaponLevel, type TrapLevel, type TroopModeStats, type UnitLevelBoostEffect, type WallLevel, type WallUpgradeEffect, type WarBaseLootEntry, type WarBaseOreEntry, type WarBonusByTownHall, type WarBonusTier, type WarLootAmount, achievements, builder, calculators, clan, clanCapital, home, magicItems, rankedBattles, seasonPass };
