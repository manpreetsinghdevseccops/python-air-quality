@Library('devseccops-shared-library@main') _

deploymentPipeline(
    // ── Core (required) ───────────────────────────────────────────────
    codeRepoUrl  : 'https://github.com/manpreetsinghdevseccops/python-air-quality',
    branchName   : 'main',
    imageName    : 'python-air-quality',
    ecrName      : 'python-air-quality',
    serviceName  : 'python-air-quality',
    awsAccount   : '130705418859',
    awsRegion    : 'ap-south-1',
    credentialsId: 'github-cred-token',

    // ── Deploy / GitOps (required when enableUpdateValuesYaml=true) ───
    organization         : 'devseccops',
    organizationManifests: '',
    valuesFile           : 'python-air-quality/values.yaml',
    configBranch         : 'dev',
    gitUsername          : '',
    gitEmail             : '',
    enableUpdateValuesYaml: false,

    // ── Build type ────────────────────────────────────────────────────
    buildType   : 'python',
    buildVersion: '3.9',

    // ── Feature flags ─────────────────────────────────────────────────
    enableBuild    : true,
    enableDeploy   : false,
    enableUnitTests: false,
    enableOwasp    : true,
    enableSonarQube: false,
)