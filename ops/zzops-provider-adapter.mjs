import { createProviderActivationPolicy } from 'zzops/provider-owner';
export default {
  sourceReference: 'polytts.owned-source', resultArtifact: 'owned-provider-result.json',
  workload() { throw new Error('Provider activation uses registered owner administration, not an inference train'); },
  verifyResult() { throw new Error('No inference qualification is asserted by provider activation'); },
  providerActivation: createProviderActivationPolicy({ app: 'polytts', service: 'polytts.service' }),
};
